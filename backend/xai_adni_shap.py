import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import torch
import shap

from backend.model import DenseNet_ViT
from backend.adni_dataset import ADNIDataset

# -----------------------
# CONFIG
# -----------------------
DEVICE = torch.device("cpu")

MODEL_PATH = "best_densenet_vit_adni.pth"
DATA_ROOT = "neuroxai_data/ADNI_SLICES"

SAVE_DIR = "results/adni/shap"
os.makedirs(SAVE_DIR, exist_ok=True)

CLASS_NAMES = ["Normal", "Mild", "Moderate"]
NUM_CLASSES = 3
NUM_SLICES = 16

# -----------------------
# LOAD MODEL
# -----------------------
model = DenseNet_ViT(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ ADNI model loaded")

# -----------------------
# LOAD DATASET
# -----------------------
dataset = ADNIDataset(root=DATA_ROOT, slices=NUM_SLICES)

# choose one subject
x, y = dataset[176]  # Mild example
x = x.unsqueeze(0).to(DEVICE)

# -----------------------
# SHAP PREDICTION FUNCTION
# -----------------------
def shap_predict(mask):
    """
    mask: (N, NUM_SLICES)
    each value indicates whether slice is kept (1) or zeroed (0)
    """
    preds = []

    for m in mask:
        x_masked = x.clone()

        for i in range(NUM_SLICES):
            if m[i] == 0:
                x_masked[:, i] = 0

        with torch.no_grad():
            out = model(x_masked)
            probs = torch.softmax(out, dim=1)

        preds.append(probs.cpu().numpy()[0])

    return np.array(preds)

# -----------------------
# SHAP EXPLAINER (SLICE-LEVEL)
# -----------------------
background = np.ones((1, NUM_SLICES))

explainer = shap.KernelExplainer(
    shap_predict,
    background
)

# explain one instance
shap_values = explainer.shap_values(
    np.ones((1, NUM_SLICES)),
    nsamples=100
)

# -----------------------
# SAVE RESULTS
# -----------------------
# -----------------------
# SAVE RESULTS (SINGLE OUTPUT)
# -----------------------
values = shap_values[0]  # shape: (NUM_SLICES,)

out_path = os.path.join(
    SAVE_DIR,
    "slice_importance.txt"
)

np.savetxt(out_path, values, fmt="%.4f")
print("✅ SHAP slice importance saved")

print("\n🎉 SHAP slice-level explanation complete")
# -----------------------
# VISUALIZE SLICE IMPORTANCE
# -----------------------
import matplotlib.pyplot as plt

slice_indices = np.arange(values.shape[0])

# predicted class
with torch.no_grad():
    pred_class = torch.argmax(model(x), dim=1).item()

values_pred = values[:, pred_class]

plt.figure(figsize=(8, 4))
plt.bar(slice_indices, values_pred)
plt.xlabel("Slice Index")
plt.ylabel("SHAP Importance")
plt.title(f"Slice-level SHAP (Predicted class: {CLASS_NAMES[pred_class]})")
plt.tight_layout()

plot_path = os.path.join(SAVE_DIR, "slice_importance_plot.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"📊 SHAP slice importance plot saved at:\n{plot_path}")
