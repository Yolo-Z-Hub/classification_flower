import os
import json
import torch
from torchvision import transforms, models
from torch import nn
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

CHECKPOINT_PATH = 'best.pt'
CAT_TO_NAME_JSON = 'cat_to_name.json'

with open(CAT_TO_NAME_JSON, 'r') as f:
    cat_to_name = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet152(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(num_ftrs, len(cat_to_name)), nn.LogSoftmax(dim=1))
state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
model.load_state_dict(state['state_dict'])
model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

#GUI
root = tk.Tk()
root.title("Flower Classifier")
root.geometry("600x500")
img_label = tk.Label(root)
img_label.pack(padx=10, pady=10)

result_var = tk.StringVar()
result_label = tk.Label(root, textvariable=result_var, font=("Arial", 16))
result_label.pack(pady=5)

def open_and_classify():
    path = filedialog.askopenfilename(
        title="选择一张图片",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    if not path:
        return

    pil_img = Image.open(path).convert("RGB")
    resized = pil_img.resize((224, 224), Image.Resampling.LANCZOS)
    tk_img = ImageTk.PhotoImage(resized)
    img_label.configure(image=tk_img)
    img_label.image = tk_img

    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.exp(output)
        idx = torch.argmax(prob, dim=1).item()
        class_label = list(cat_to_name.keys())[idx]
        name = cat_to_name[class_label]
        result_var.set(f"预测结果：{name}")

btn = tk.Button(root, text="打开图片并预测", command=open_and_classify)
btn.pack(pady=10)

root.mainloop()
