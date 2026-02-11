import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use("Agg")   # 👈 non-GUI backend

import matplotlib.pyplot as plt

from backend.model import DenseNet_ViT
from backend.dataset import OASISDataset

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = torch.device("cpu")  # keep CPU for stability
NUM_CLASSES = 3
BATCH_SIZE = 1
EPOCHS = 8
LR = 1e-4
SLICES = 16

CLASS_NAMES = ["Normal", "Mild", "Moderate"]

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results", "oasis", "training")
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# DATASET
# -----------------------------
dataset = OASISDataset(slices=SLICES)
print(f"[INFO] Loaded {len(dataset)} subjects")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"[INFO] Training samples: {len(train_ds)}")
print(f"[INFO] Validation samples: {len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

# -----------------------------
# MODEL
# -----------------------------
model = DenseNet_ViT(num_classes=NUM_CLASSES).to(DEVICE)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)
criterion = nn.CrossEntropyLoss()

# -----------------------------
# TRACKING
# -----------------------------
train_losses = []
train_accs = []
val_accs = []

best_val_acc = 0.0
best_epoch = -1

# -----------------------------
# TRAINING LOOP
# -----------------------------
for epoch in range(EPOCHS):
    # ---- TRAIN ----
    model.train()
    correct, total = 0, 0
    running_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    # ---- VALIDATION ----
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    val_acc = correct / total

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.4f} "
        f"Val Acc: {val_acc:.4f}"
    )

    # ---- SAVE BEST MODEL ----
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1

        torch.save(
            {
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": best_val_acc,
                "num_classes": NUM_CLASSES,
                "slices": SLICES
            },
            os.path.join(RESULTS_DIR, "best_model.pth")
        )

        print(
            f"✅ Saved BEST model at epoch {best_epoch} "
            f"(Val Acc = {best_val_acc:.4f})"
        )

# -----------------------------
# PLOTS
# -----------------------------
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve (OASIS)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"))
plt.close()

plt.figure()
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve (OASIS)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "accuracy_curve.png"))
plt.close()

# -----------------------------
# SAVE FINAL METRICS
# -----------------------------
with open(os.path.join(RESULTS_DIR, "final_metrics.txt"), "w") as f:
    f.write("FINAL TRAINING RESULTS – OASIS\n\n")
    f.write(f"Best Epoch: {best_epoch}\n")
    f.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")
    f.write(f"Epochs Trained: {EPOCHS}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Slices per Subject: {SLICES}\n")

print("\n===== TRAINING COMPLETE =====")
print(f"Best Epoch: {best_epoch}")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")
print(f"Saved to: {RESULTS_DIR}")
