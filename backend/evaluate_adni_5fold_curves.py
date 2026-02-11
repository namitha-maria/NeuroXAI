import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

from backend.model import DenseNet_ViT
from backend.adni_dataset import ADNIDataset

# ----------------------------
# CONFIG
# ----------------------------
DEVICE = torch.device("cpu")
NUM_CLASSES = 3
CLASS_NAMES = ["Normal", "Mild", "Moderate"]

N_SPLITS = 5
EPOCHS = 10
BATCH_SIZE = 8
LR = 1e-4
NUM_SLICES = 16

SAVE_DIR = "results/adni/5fold_training_curves"
os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------------
# DATASET
# ----------------------------
dataset = ADNIDataset(
    root="neuroxai_data/ADNI_SLICES",
    slices=NUM_SLICES
)

labels = np.array([int(dataset[i][1]) for i in range(len(dataset))])

print(f"[INFO] Loaded {len(dataset)} ADNI subjects")

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=42
)

# ----------------------------
# STORAGE FOR CURVES
# ----------------------------
all_train_acc = []
all_val_acc = []
all_train_loss = []
all_val_loss = []

# ----------------------------
# 5-FOLD TRAINING
# ----------------------------
for fold, (train_idx, val_idx) in enumerate(
    skf.split(np.zeros(len(labels)), labels)
):
    print(f"\n==============================")
    print(f"===== Fold {fold+1}/{N_SPLITS} =====")
    print(f"==============================")

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = DenseNet_ViT(num_classes=NUM_CLASSES).to(DEVICE)

    class_weights = torch.tensor([1.0, 2.0, 3.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_acc, val_acc = [], []
    train_loss, val_loss = [], []

    # ------------------------
    # EPOCH LOOP
    # ------------------------
    for epoch in range(EPOCHS):
        print(f"\n[Fold {fold+1}] Epoch {epoch+1}/{EPOCHS} — TRAIN")

        model.train()
        correct, total, running_loss = 0, 0, 0.0

        train_bar = tqdm(
            train_loader,
            desc=f"Fold {fold+1} | Train",
            leave=False
        )

        for batch_idx, (x, y) in enumerate(train_bar):
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

            train_bar.set_postfix(loss=f"{loss.item():.4f}")

            # heartbeat every ~30 batches
            if batch_idx % 30 == 0:
                print(
                    f"[Fold {fold+1} | Epoch {epoch+1}] "
                    f"Processed batch {batch_idx}/{len(train_loader)}"
                )

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = correct / total

        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        # ------------------------
        # VALIDATION
        # ------------------------
        print(f"[Fold {fold+1}] Epoch {epoch+1}/{EPOCHS} — VALIDATION")

        model.eval()
        correct, total, running_loss = 0, 0, 0.0

        val_bar = tqdm(
            val_loader,
            desc=f"Fold {fold+1} | Val",
            leave=False
        )

        with torch.no_grad():
            for x, y in val_bar:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = criterion(out, y)

                running_loss += loss.item()
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)

        epoch_val_loss = running_loss / len(val_loader)
        epoch_val_acc = correct / total

        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)

        print(
            f"[Fold {fold+1}] Epoch {epoch+1} DONE | "
            f"Train Acc: {epoch_train_acc:.4f} | "
            f"Val Acc: {epoch_val_acc:.4f}"
        )

    all_train_acc.append(train_acc)
    all_val_acc.append(val_acc)
    all_train_loss.append(train_loss)
    all_val_loss.append(val_loss)

# ----------------------------
# PLOT: ACCURACY (ALL FOLDS)
# ----------------------------
plt.figure(figsize=(8, 5))

for i in range(N_SPLITS):
    plt.plot(
        all_train_acc[i],
        label=f"Fold {i+1} - Training Accuracy"
    )
    plt.plot(
        all_val_acc[i],
        linestyle="--",
        label=f"Fold {i+1} - Validation Accuracy"
    )

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy (5-Fold CV)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(
    os.path.join(SAVE_DIR, "accuracy_5fold.png"),
    dpi=300
)
plt.close()

# ----------------------------
# PLOT: LOSS (ALL FOLDS)
# ----------------------------
plt.figure(figsize=(8, 5))

for i in range(N_SPLITS):
    plt.plot(
        all_train_loss[i],
        label=f"Fold {i+1} - Training Loss"
    )
    plt.plot(
        all_val_loss[i],
        linestyle="--",
        label=f"Fold {i+1} - Validation Loss"
    )

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss (5-Fold CV)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(
    os.path.join(SAVE_DIR, "loss_5fold.png"),
    dpi=300
)
plt.close()

print("\n✅ 5-fold accuracy & loss curves saved")
print("Saved to:", SAVE_DIR)
