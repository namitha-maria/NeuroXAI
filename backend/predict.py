import os
import cv2
import torch
import numpy as np

from .models import DenseNet_ViT

device = "cpu"
SLICE_COUNT = 16
CLASS_NAMES = ["Normal", "Mild", "Moderate"]

model = DenseNet_ViT().to(device)
MODEL_PATH = "../models/best_densenet_vit_oasis.pth"
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model.eval()

subject_path = "neuroxai_data/Moderate/OAS1_0031_MR1"  # CHANGE THIS

imgs = []
slice_files = sorted(os.listdir(subject_path))
start = (len(slice_files) - SLICE_COUNT) // 2
slice_files = slice_files[start:start + SLICE_COUNT]

for f in slice_files:
    img = cv2.imread(os.path.join(subject_path, f), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.stack([img, img, img], axis=0)
    imgs.append(img)

x = torch.tensor(np.array(imgs), dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    out = model(x)
    probs = torch.softmax(out, dim=1)
    pred = probs.argmax(dim=1).item()

print("\nPrediction Results")
print("------------------")
for i, c in enumerate(CLASS_NAMES):
    print(f"{c}: {probs[0][i]*100:.2f}%")
print("------------------")
print("Final Prediction:", CLASS_NAMES[pred])
