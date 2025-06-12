import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets, models
from torch import nn

DATA_DIR = './flower_data/'
VALID_DIR = os.path.join(DATA_DIR, 'valid')
CHECKPOINT_PATH = 'best.pt'
CAT_TO_NAME_JSON = 'cat_to_name.json'
BATCH_SIZE = 8
FORCE_CPU = False

with open(CAT_TO_NAME_JSON, 'r') as f:
    cat_to_name = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() and not FORCE_CPU else 'cpu')
print(f'Using device: {device}')

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
val_dataset = datasets.ImageFolder(VALID_DIR, transform=val_transform)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True)

model = models.resnet152(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(num_ftrs, len(val_dataset.classes)),
                         nn.LogSoftmax(dim=1))
state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
model.load_state_dict(state['state_dict'])
model.to(device)
model.eval()

#反标准化函数
def im_convert(tensor):
    img = tensor.cpu().clone().detach().numpy().transpose(1,2,0)
    img = img * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])
    img = np.clip(img, 0, 1)
    return img

# 批量预测
dataiter = iter(val_loader)
images, labels = next(dataiter)
images_device = images.to(device)

with torch.no_grad():
    outputs = model(images_device)
    probs = torch.exp(outputs)
    preds = torch.argmax(probs, dim=1)

fig = plt.figure(figsize=(20, 20))
columns = 4
rows = 2
for idx in range(columns * rows):
    ax = fig.add_subplot(rows, columns, idx+1, xticks=[], yticks=[])
    img = im_convert(images[idx])
    ax.imshow(img)

    pred_label = val_dataset.classes[preds[idx]]
    true_label = val_dataset.classes[labels[idx].item()]
    pred_name = cat_to_name.get(pred_label, pred_label)
    true_name = cat_to_name.get(true_label, true_label)
    color = 'green' if pred_name == true_name else 'red'
    ax.set_title(f"{pred_name} ({true_name})", color=color)
plt.tight_layout()
plt.subplots_adjust(hspace=0.2)
plt.show()

