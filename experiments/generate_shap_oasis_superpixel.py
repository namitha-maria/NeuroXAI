# training/generate_shap_oasis_superpixel.py

import os
import numpy as np
import torch
import shap
import cv2
import matplotlib
matplotlib.use("Agg")  # IMPORTANT: no GUI / no Tkinter
import matplotlib.pyplot as plt

from skimage.segmentation import slic
from skimage.color import gray2rgb

from backend.model import DenseNet_ViT
from backend.dataset import OASISDataset

# -----------------------
# CONFIG
# -----------------------
DEVICE = torch.device("cpu")

SAVE_DIR = "results/oasis/shap_superpixel"
os.makedirs(SAVE_DIR, exist_ok=True)

CLASS_NAMES = ["Normal", "Mild", "Moderate"]
NUM_CLASSES = 3

NUM_SUPERPIXELS = 40        # keep 30–60 for paper
NSAMPLES = 50               # small for CPU safety
SLICE_COUNT = 16

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

# -----------------------
# LOAD DATASET
# -----------------------
dataset = OASISDataset(slices=SLICE_COUNT)
print(f"[INFO] Loaded {len(dataset)} subjects")

# pick one representative subject per class
selected_subjects = {}
for idx in range(len(dataset)):
    _, y = dataset[idx]
    if int(y) not in selected_subjects:
        selected_subjects[int(y)] = idx
    if len(selected_subjects) == NUM_CLASSES:
        break

print("[INFO] Selected subjects:", selected_subjects)

# -----------------------
# LOOP OVER CLASSES
# -----------------------
for cls, subject_idx in selected_subjects.items():

    print(f"\n🔍 SHAP for class: {CLASS_NAMES[cls]}")

    x, y = dataset[subject_idx]
    x = x.unsqueeze(0).to(DEVICE)

    # take middle slice
    slice_idx = x.shape[1] // 2
    slice_tensor = x[:, slice_idx]      # (1, 3, H, W)

    # to numpy image
    image = slice_tensor[0].permute(1, 2, 0).cpu().numpy()
    image = np.clip(image, 0, 1)

    if image.shape[2] == 1:
        image = gray2rgb(image)

    # -----------------------
    # SUPERPIXELS
    # -----------------------
    segments = slic(
        image,
        n_segments=NUM_SUPERPIXELS,
        compactness=10,
        sigma=1,
        start_label=0
    )

    num_features = np.unique(segments).shape[0]

    # -----------------------
    # MODEL PRED FUNCTION
    # -----------------------
    def predict_from_mask(mask):
        """
        mask: (N, num_superpixels)
        """
        imgs = np.zeros((mask.shape[0], *image.shape), dtype=np.float32)

        for i, m in enumerate(mask):
            img = image.copy()
            for sp in range(num_features):
                if m[sp] == 0:
                    img[segments == sp] = 0
            imgs[i] = img

        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2)
        imgs = imgs.unsqueeze(1).repeat(1, SLICE_COUNT, 1, 1, 1)

        with torch.no_grad():
            out = model(imgs)
            probs = torch.softmax(out, dim=1)

        return probs.cpu().numpy()

    # -----------------------
    # SHAP EXPLAINER
    # -----------------------
    explainer = shap.KernelExplainer(
        predict_from_mask,
        np.zeros((1, num_features))
    )

    shap_values = explainer.shap_values(
        np.ones((1, num_features)),
        nsamples=NSAMPLES
    )

    # -----------------------
    # VISUALIZATION
    # -----------------------
    fig, axes = plt.subplots(1, NUM_CLASSES + 1, figsize=(16, 4))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    for c in range(NUM_CLASSES):
        shap_map = np.zeros(segments.shape)

        for sp in range(num_features):
            shap_map[segments == sp] = shap_values[c][0][sp]

        axes[c + 1].imshow(image, cmap="gray")
        im = axes[c + 1].imshow(
            shap_map,
            cmap="RdBu_r",
            alpha=0.6
        )
        axes[c + 1].set_title(CLASS_NAMES[c])
        axes[c + 1].axis("off")

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    plt.tight_layout()

    save_path = os.path.join(
        SAVE_DIR,
        f"shap_superpixel_{CLASS_NAMES[cls]}.png"
    )
    plt.savefig(save_path, dpi=200)
    plt.close()

    # -----------------------
    # LOG PREDICTION
    # -----------------------
    with torch.no_grad():
        pred = model(x)
        pred_cls = torch.argmax(pred, dim=1).item()

    print(f"✅ SHAP saved for {CLASS_NAMES[cls]}")
    print(f"   True label: {CLASS_NAMES[y]}")
    print(f"   Predicted: {CLASS_NAMES[pred_cls]}")
