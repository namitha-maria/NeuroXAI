# training/generate_shap_oasis.py

import os
import numpy as np
import torch
import shap

from backend.model import DenseNet_ViT
from backend.dataset import OASISDataset

# -----------------------
# CONFIG
# -----------------------
DEVICE = torch.device("cpu")
SAVE_DIR = "results/oasis/shap"
os.makedirs(SAVE_DIR, exist_ok=True)

CLASS_NAMES = ["Normal", "Mild", "Moderate"]

SUBJECT_INDICES = {
    0: 0,
    1: 135,
    2: 205
}

# -----------------------
# LOAD MODEL
# -----------------------
model = DenseNet_ViT(num_classes=3).to(DEVICE)
ckpt = torch.load("results/oasis/training/best_model.pth", map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# -----------------------
# DATASET
# -----------------------
dataset = OASISDataset(slices=16)

# -----------------------
# FEATURE EXTRACTION (EXACT MODEL PIPELINE)
# -----------------------
def extract_latent_features(x):
    """
    x: (1, 16, 3, 224, 224)
    returns: (1, 256)  ← IMPORTANT
    """
    with torch.no_grad():
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)

        feats = model.cnn(x)                 # (B*S, 1024)
        feats = feats.view(B, S, -1).mean(1) # (B, 1024)

        feats = model.proj(feats)            # (B, 256)
    return feats.cpu().numpy()

# -----------------------
# CLASSIFIER WRAPPER
# -----------------------
def classifier_fn(z):
    """
    z: (N, 256)
    returns: (N, num_classes)
    """
    z = torch.tensor(z).float()
    with torch.no_grad():
        out = model.classifier(z)
        probs = torch.softmax(out, dim=1)
    return probs.numpy()

# -----------------------
# BACKGROUND (VERY SMALL)
# -----------------------
x0, _ = dataset[0]
x0 = x0.unsqueeze(0)
background = extract_latent_features(x0)

# -----------------------
# SHAP EXPLAINER
# -----------------------
explainer = shap.KernelExplainer(
    classifier_fn,
    background
)
# -----------------------
# RUN SHAP
# -----------------------
for cls, idx in SUBJECT_INDICES.items():
    print(f"\n🔍 SHAP for class: {CLASS_NAMES[cls]}")

    x, y = dataset[idx]
    x = x.unsqueeze(0)

    latent = extract_latent_features(x)

    shap_values = explainer.shap_values(
        latent,
        nsamples=50
    )

    probs = classifier_fn(latent)[0]
    pred_class = np.argmax(probs)

    np.save(
        f"{SAVE_DIR}/shap_{CLASS_NAMES[pred_class]}.npy",
        shap_values[0]
    )

    print(f"✅ SHAP saved for {CLASS_NAMES[pred_class]}")
    print(f"   True label: {CLASS_NAMES[y]}")
    print(f"   Predicted: {CLASS_NAMES[pred_class]}")
