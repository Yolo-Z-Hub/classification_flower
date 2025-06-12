import os
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import time
import copy

# 全局均值和标准差，用于反归一化保存图片
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

DATA_DIR = './flower_data/'
MODEL_NAME_CONFIG = 'resnet'
NUM_CLASSES_CONFIG = 102
FEATURE_EXTRACT_CONFIG = True  # 阶段1只训练最后一层
BATCH_SIZE_CONFIG = 8
NUM_EPOCHS_CONFIG = 20
FINETUNE_EPOCHS = 10
FILENAME_CHECKPOINT_CONFIG = 'finetuned.pt'
FILENAME_FINETUNED = 'best.pt'
LOCAL_RESNET152_WEIGHTS_PATH = r'E:\ccnu_project\resnet152-394f9c45.pth'
VERBOSE = True

def train_model(model, dataloaders, criterion, optimizer, scheduler,
                num_epochs, device, filename,writer=None, save_img_dir=None):
    since = time.time()
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if writer is not None:
                writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch + 1)
                writer.add_scalar(f'Acc/{phase}', epoch_acc, epoch + 1)

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                }, filename)
                print(f"Saved best model: acc={best_acc:.4f}")

        scheduler.step()

        # 可视化训练集图像
        if save_img_dir is not None:
            os.makedirs(save_img_dir, exist_ok=True)
            inputs_viz, _ = next(iter(dataloaders['train']))
            img_grid = (inputs_viz[:8] * STD + MEAN).clamp(0, 1)
            save_path = os.path.join(save_img_dir, f'epoch_{epoch+1:02d}.png')
            save_image(img_grid, save_path, nrow=4)
            print(f"Saved training images to {save_path}")

    time_elapsed = time.time() - since
    print(f'Training complete in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

def main():
    train_dir = os.path.join(DATA_DIR, 'train')
    valid_dir = os.path.join(DATA_DIR, 'valid')

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(45),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(0.2, 0.1, 0.1, 0.1),
            transforms.RandomGrayscale(p=0.025),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
        for x in ['train', 'valid']
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=BATCH_SIZE_CONFIG,
            shuffle=True,
            num_workers=4
        )
        for x in ['train', 'valid']
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    log_dir = 'runs/flower_experiment'
    writer = SummaryWriter(log_dir)

    # 定义辅助函数：冻结参数
    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def initialize_model_local(model_name_arg, num_classes_arg, feature_extract_arg, local_weights_path_arg=None, device_arg='cpu'):
        model_ft = None
        if model_name_arg == "resnet":
            model_ft = models.resnet152(weights=None)

            if local_weights_path_arg and os.path.exists(local_weights_path_arg):
                try:
                    state_dict = torch.load(local_weights_path_arg, map_location=device_arg, weights_only=True)
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] if k.startswith('module.') else k
                        new_state_dict[name] = v
                    model_ft.load_state_dict(new_state_dict)
                except Exception as e:
                    print(f"Failed to load weights: {e}")
            else:
                print("Local weights not found, using random init.")
            # 冻结参数
            set_parameter_requires_grad(model_ft, feature_extract_arg)
            # 改造最后一层分类器
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(512, num_classes_arg),
                nn.LogSoftmax(dim=1)
            )

        return model_ft, 224

    model, _ = initialize_model_local(
        MODEL_NAME_CONFIG,
        NUM_CLASSES_CONFIG,
        FEATURE_EXTRACT_CONFIG,
        LOCAL_RESNET152_WEIGHTS_PATH if MODEL_NAME_CONFIG == 'resnet' else None,
        device
    )
    if model is None:
        print("Model init failed.")
        return
    model = model.to(device)

    # 阶段1：特征提取
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    if VERBOSE:
        print(f"Stage1 Params to update: {len(params_to_update)}")

    optimizer = optim.Adam(params_to_update, lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.NLLLoss()

    print("\n--- Stage 1: Feature-extract training ---")
    train_model(
        model, dataloaders, criterion, optimizer, scheduler,
        NUM_EPOCHS_CONFIG, device, FILENAME_CHECKPOINT_CONFIG,
        writer=writer,
        save_img_dir='saved_images/stage1'
    )

    # 阶段2：微调全部参数
    for param in model.parameters():
        param.requires_grad = True
    if VERBOSE:
        print("Stage2: Unfroze all parameters.")

    optimizer_finetune = optim.Adam(model.parameters(), lr=1e-4)
    scheduler_finetune = optim.lr_scheduler.StepLR(optimizer_finetune, step_size=7, gamma=0.1)

    chk = torch.load(FILENAME_CHECKPOINT_CONFIG, map_location=device)
    model.load_state_dict(chk['state_dict'])
    if VERBOSE:
        print(f"Loaded best model from {FILENAME_CHECKPOINT_CONFIG}, acc={chk['best_acc']:.4f}")

    print("\n--- Stage 2: Fine-tuning all layers ---")
    train_model(
        model, dataloaders, criterion,
        optimizer_finetune, scheduler_finetune,
        FINETUNE_EPOCHS, device, FILENAME_FINETUNED,
        writer=writer,
        save_img_dir='saved_images/stage2'
    )

    writer.close()

if __name__ == '__main__':
    main()
