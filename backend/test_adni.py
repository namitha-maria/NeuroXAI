import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)

from backend.adni_dataset import ADNIDataset
from backend.model import DenseNet_ViT

# ----------------------------
# DEVICE
# ----------------------------
device = "cpu"
print("Using device:", device)

# ----------------------------
# CONFIG
# ----------------------------
NUM_CLASSES = 3
BATCH_SIZE = 1
NUM_SLICES = 16
MODEL_PATH = "best_densenet_vit_adni.pth"
CLASS_NAMES = ["Normal", "Mild", "Moderate"]

# ----------------------------
# DATASET
# ----------------------------
dataset = ADNIDataset(
    root="neuroxai_data/ADNI_SLICES",
    slices=NUM_SLICES
)

train_size = int(0.85 * len(dataset))
test_size = len(dataset) - train_size

_, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size]
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"[INFO] Test subjects: {len(test_dataset)}")

# ----------------------------
# MODEL
# ----------------------------
model = DenseNet_ViT(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ----------------------------
# TESTING
# ----------------------------
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        y_true.append(labels.item())
        y_pred.append(preds.item())

# ----------------------------
# METRICS
# ----------------------------
acc = accuracy_score(y_true, y_pred)
bal_acc = balanced_accuracy_score(y_true, y_pred)

print("\n================ ADNI TEST RESULTS ================")
print(f"Accuracy: {acc:.4f}")
print(f"Balanced Accuracy: {bal_acc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )
)

print("===================================================")
