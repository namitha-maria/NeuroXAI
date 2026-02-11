import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

from backend.model import DenseNet_ViT
from backend.adni_dataset import ADNIDataset

# ----------------------------
# CONFIG
# ----------------------------
DEVICE = "cpu"
MODEL_PATH = "best_densenet_vit_adni.pth"
DATA_ROOT = "neuroxai_data/ADNI_SLICES"

CLASS_NAMES = ["Normal", "Mild", "Moderate"]
NUM_CLASSES = 3
BATCH_SIZE = 8

SAVE_DIR = "results/adni/evaluation"
import os
os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------------
# LOAD MODEL
# ----------------------------
model = DenseNet_ViT(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ Model loaded")

# ----------------------------
# LOAD DATASET (TEST SPLIT)
# ----------------------------
dataset = ADNIDataset(root=DATA_ROOT, slices=16)

test_size = int(0.15 * len(dataset))
train_size = len(dataset) - test_size

_, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size]
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"[INFO] Test samples: {len(test_dataset)}")

# ----------------------------
# RUN INFERENCE
# ----------------------------
y_true = []
y_pred = []
y_prob = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

# ----------------------------
# CONFUSION MATRIX
# ----------------------------
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(5, 5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(range(NUM_CLASSES), CLASS_NAMES)
plt.yticks(range(NUM_CLASSES), CLASS_NAMES)

for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"), dpi=300)
plt.close()

# ----------------------------
# CLASSIFICATION REPORT
# ----------------------------
report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES
)

print("\nClassification Report:")
print(report)

with open(os.path.join(SAVE_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# ----------------------------
# ROC-AUC (MULTICLASS)
# ----------------------------
# binarize labels
y_true_bin = label_binarize(y_true, classes=[0, 1, 2])

fpr = {}
tpr = {}
roc_auc = {}

for i in range(NUM_CLASSES):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# ----------------------------
# PLOT ROC CURVES
# ----------------------------
plt.figure(figsize=(6, 5))

for i in range(NUM_CLASSES):
    plt.plot(
        fpr[i],
        tpr[i],
        label=f"{CLASS_NAMES[i]} (AUC = {roc_auc[i]:.3f})"
    )

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (One-vs-Rest)")
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(SAVE_DIR, "roc_curves.png"), dpi=300)
plt.close()

print("\n✅ Evaluation complete")
print("Saved to:", SAVE_DIR)
