import matplotlib
matplotlib.use("Agg")  # non-GUI backend (Windows-safe)

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from backend.model import DenseNet_ViT
from backend.adni_dataset import ADNIDataset
from backend.gradcam_utils import GradCAM

# ----------------------------
# CONFIG
# ----------------------------
DEVICE = "cpu"
MODEL_PATH = "best_densenet_vit_adni.pth"
DATA_ROOT = "neuroxai_data/ADNI_SLICES"

CLASS_NAMES = ["Normal", "Mild", "Moderate"]
NUM_CLASSES = 3
SLICE_COUNT = 16
IMG_SIZE = 224

OUTPUT_DIR = "results/xai/adni"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Using device:", DEVICE)

# ----------------------------
# LOAD MODEL
# ----------------------------
model = DenseNet_ViT(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ----------------------------
# INIT GRADCAM
# ----------------------------
gradcam = GradCAM(
    model=model,
    target_layer=model.cnn.features.denseblock4
)

# ----------------------------
# LOAD DATASET
# ----------------------------
dataset = ADNIDataset(
    root=DATA_ROOT,
    slices=SLICE_COUNT
)

print(f"[INFO] Loaded {len(dataset)} ADNI subjects")

# ----------------------------
# SELECT ONE SUBJECT PER CLASS
# ----------------------------
selected_indices = {}

for idx in range(len(dataset)):
    _, y = dataset[idx]
    label = y.item()

    if label not in selected_indices:
        selected_indices[label] = idx

    if len(selected_indices) == NUM_CLASSES:
        break

print("\nSelected subjects:")
for cls, idx in selected_indices.items():
    print(f"{CLASS_NAMES[cls]} → subject index {idx}")

# ----------------------------
# HELPER: VENTRICULAR AXIAL SLICE
# ----------------------------
def select_ventricular_slice(imgs):
    """
    Select axial slice likely at ventricular level
    by detecting central low-intensity CSF.
    """
    scores = []

    for i, img in enumerate(imgs):
        gray = img[0]
        h, w = gray.shape

        center = gray[h//3:2*h//3, w//3:2*w//3]
        csf_score = np.mean(center < 0.25)

        scores.append((csf_score, i))

    return max(scores)[1]

# ----------------------------
# GENERATE GRADCAM PER CLASS
# ----------------------------
for class_id, subject_idx in selected_indices.items():
    x, y = dataset[subject_idx]

    # add batch dimension
    x = x.unsqueeze(0).to(DEVICE)

    # ----------------------------
    # MODEL PREDICTION
    # ----------------------------
    with torch.no_grad():
        outputs = model(x)
        pred = torch.argmax(outputs, dim=1).item()

    print(
        f"\n[INFO] {CLASS_NAMES[class_id]} subject {subject_idx} "
        f"(predicted: {CLASS_NAMES[pred]})"
    )

    # ----------------------------
    # GRADCAM GENERATION
    # ----------------------------
    cams = gradcam.generate(x, pred)  # (S, H, W)

    imgs = x[0].cpu().numpy()          # (S, 3, H, W)

    # select ventricular slice
    slice_idx = select_ventricular_slice(imgs)

    # OPTION 2: move slice upward for clarity
    slice_idx = min(slice_idx + 2, imgs.shape[0] - 1)

    print(f"[INFO] Using slice index: {slice_idx}")

    # ----------------------------
    # PREPARE ORIGINAL SLICE
    # ----------------------------
    orig_slice = imgs[slice_idx][0]
    orig_slice = (orig_slice * 255).astype(np.uint8)
    orig_slice = cv2.cvtColor(orig_slice, cv2.COLOR_GRAY2BGR)

    # ----------------------------
    # PREPARE HEATMAP
    # ----------------------------
    cam = cams[slice_idx]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = (cam * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    # ----------------------------
    # OPTION 1: BRAIN MASK (SAFE)
    # ----------------------------
    gray = cv2.cvtColor(orig_slice, cv2.COLOR_BGR2GRAY)

    # binary brain mask
    _, brain_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)

    # FORCE mask to match heatmap exactly
    brain_mask = cv2.resize(brain_mask, (heatmap.shape[1], heatmap.shape[0]))
    brain_mask = brain_mask.astype(np.uint8)

    # apply mask safely
    heatmap = cv2.bitwise_and(heatmap, heatmap, mask=brain_mask)


    # ----------------------------
    # FINAL SAFE OVERLAY (BULLETPROOF)
    # ----------------------------

    # Force orig_slice to canonical OpenCV format
    orig_slice_safe = cv2.resize(orig_slice, (IMG_SIZE, IMG_SIZE))
    orig_slice_safe = orig_slice_safe.astype(np.uint8)
    orig_slice_safe = np.ascontiguousarray(orig_slice_safe)

    # Force heatmap to canonical OpenCV format
    heatmap_safe = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_safe = heatmap_safe.astype(np.uint8)
    heatmap_safe = np.ascontiguousarray(heatmap_safe)

    # Double-check channels (must both be 3)
    if orig_slice_safe.ndim == 2:
        orig_slice_safe = cv2.cvtColor(orig_slice_safe, cv2.COLOR_GRAY2BGR)
    if heatmap_safe.ndim == 2:
        heatmap_safe = cv2.cvtColor(heatmap_safe, cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(
        orig_slice_safe, 0.6,
        heatmap_safe, 0.4,
        0
    )


    # ----------------------------
    # SAVE RESULTS
    # ----------------------------
    save_dir = os.path.join(
        OUTPUT_DIR,
        f"class_{CLASS_NAMES[class_id]}_subject_{subject_idx}"
    )
    os.makedirs(save_dir, exist_ok=True)

    cv2.imwrite(os.path.join(save_dir, "original.png"), orig_slice)
    cv2.imwrite(os.path.join(save_dir, "heatmap.png"), heatmap)
    cv2.imwrite(os.path.join(save_dir, "overlay.png"), overlay)

    print(f"[INFO] Saved Grad-CAM to {save_dir}")

print("\n✅ Grad-CAM generation complete for all classes")
