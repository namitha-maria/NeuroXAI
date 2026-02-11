import torch
import numpy as np
import os
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from .dataset import OASISDataset
from .model import DenseNet_ViT

# ----------------------------
# DEVICE
# ----------------------------
device = "cpu"
print("Using device:", device)

# ----------------------------
# CONFIG
# ----------------------------
NUM_SLICES = 16
CLASS_NAMES = ["Normal", "Mild", "Moderate"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "best_densenet_vit_oasis.pth"
)   # 👈 IMPORTANT

# ----------------------------
# DATASET
# ----------------------------
dataset = OASISDataset(slices=NUM_SLICES)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

_, _, test_ds = random_split(dataset, [train_size, val_size, test_size])

test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

print(f"[INFO] Test samples: {len(test_ds)}")

# ----------------------------
# MODEL
# ----------------------------
model = DenseNet_ViT().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ----------------------------
# EVALUATION
# ----------------------------
y_true = []
y_pred = []
y_probs = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        out = model(x)
        probs = torch.softmax(out, dim=1)

        pred = probs.argmax(dim=1)

        y_true.append(y.item())
        y_pred.append(pred.item())
        y_probs.append(probs.cpu().numpy()[0])

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_probs = np.array(y_probs)

# ----------------------------
# METRICS
# ----------------------------
acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4
)

# Class-wise accuracy
class_acc = cm.diagonal() / cm.sum(axis=1)

# ROC-AUC (multi-class, One-vs-Rest)
y_true_oh = np.eye(len(CLASS_NAMES))[y_true]
roc_auc = roc_auc_score(y_true_oh, y_probs, multi_class="ovr")

# ----------------------------
# PRINT RESULTS
# ----------------------------
print("\n================ TEST RESULTS ================")
print(f"Overall Test Accuracy: {acc * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

print("\nClass-wise Accuracy:")
for i, cls in enumerate(CLASS_NAMES):
    print(f"{cls}: {class_acc[i] * 100:.2f}%")

print("\nClassification Report:")
print(report)

print(f"ROC-AUC (OvR): {roc_auc:.4f}")
print("==============================================")
