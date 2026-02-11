# training/generate_gradcam_oasis.py

import os
import torch
import cv2
import numpy as np

from backend.model import DenseNet_ViT
from backend.dataset import OASISDataset
from training.gradcam_dense import GradCAMDense

# -----------------------
# CONFIG
# -----------------------
DEVICE = torch.device("cpu")
NUM_CLASSES = 3
CLASS_NAMES = ["Normal", "Mild", "Moderate"]

BASE_SAVE_DIR = "results/oasis/gradcam"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

# -----------------------
# LOAD MODEL
# -----------------------
model = DenseNet_ViT(num_classes=NUM_CLASSES).to(DEVICE)

ckpt = torch.load(
    "results/oasis/training/best_model.pth",
    map_location=DEVICE
)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("✅ Model loaded")

# -----------------------
# INIT GRADCAM
# -----------------------
target_layer = model.cnn.features.denseblock4
gradcam = GradCAMDense(model, target_layer)

# -----------------------
# LOAD DATASET
# -----------------------
dataset = OASISDataset(slices=16)
print(f"[INFO] Loaded {len(dataset)} subjects")

# -----------------------
# SELECT ONE SUBJECT PER CLASS
# -----------------------
selected_indices = {}

for idx in range(len(dataset)):
    _, y = dataset[idx]
    label = y.item()

    if label not in selected_indices:
        selected_indices[label] = idx

    if len(selected_indices) == NUM_CLASSES:
        break

print("\nSelected subjects per class:")
for cls, idx in selected_indices.items():
    print(f"{CLASS_NAMES[cls]} → subject index {idx}")

# -----------------------
# GENERATE GRADCAM
# -----------------------
for class_id, subject_idx in selected_indices.items():
    x, y = dataset[subject_idx]

    # add batch dimension
    x = x.unsqueeze(0).to(DEVICE)

    # take middle slice
    slice_idx = x.shape[1] // 2
    slice_tensor = x[:, slice_idx]   # (1, 3, 224, 224)

    # generate gradcam
    cam = gradcam.generate(slice_tensor)

    # -----------------------
    # SAVE RESULTS
    # -----------------------
    save_dir = os.path.join(
        BASE_SAVE_DIR,
        f"class_{CLASS_NAMES[class_id]}_subject_{subject_idx}"
    )
    os.makedirs(save_dir, exist_ok=True)

    # original image
    orig = slice_tensor[0].permute(1, 2, 0).cpu().numpy()
    orig = np.clip(orig, 0, 1)
    orig = (orig * 255).astype(np.uint8)

    # heatmap
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    # overlay
    overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

    cv2.imwrite(os.path.join(save_dir, "original.png"), orig)
    cv2.imwrite(os.path.join(save_dir, "heatmap.png"), heatmap)
    cv2.imwrite(os.path.join(save_dir, "overlay.png"), overlay)

    print(
        f"✅ Saved Grad-CAM for {CLASS_NAMES[class_id]} "
        f"(subject {subject_idx}, true label = {CLASS_NAMES[y.item()]})"
    )

print("\n🎉 Grad-CAM generation complete for all classes")
