# =========================================================
# Cross-Validation Script (5-Fold)
# DenseNet–ViT | OASIS Dataset
# =========================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU

# -----------------------------
# Matplotlib (NON-GUI backend)
# -----------------------------
import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from backend.model import DenseNet_ViT
from backend.dataset import OASISDataset

# =========================================================
# CONFIG
# =========================================================
DEVICE = torch.device("cpu")
NUM_CLASSES = 3
BATCH_SIZE = 1
EPOCHS = 1              # Increase if retraining
LR = 1e-4
N_SPLITS = 5
NUM_SLICES = 16

CLASS_NAMES = ["Normal", "Mild", "Moderate"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================================================
# DATASET
# =========================================================
dataset = OASISDataset(slices=NUM_SLICES)
print(f"[INFO] Loaded {len(dataset)} subjects")

# =========================================================
# K-FOLD SETUP
# =========================================================
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

all_metrics = []

# =========================================================
# CROSS-VALIDATION LOOP
# =========================================================
for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    print(f"\n===== Fold {fold+1}/{N_SPLITS} =====")

    fold_dir = os.path.join(RESULTS_DIR, f"fold_{fold+1}")
    os.makedirs(fold_dir, exist_ok=True)

    train_ds = Subset(dataset, train_idx)
    test_ds = Subset(dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

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
    # TRAINING
    # -----------------------------
    model.train()
    for epoch in range(EPOCHS):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    # -----------------------------
    # TESTING
    # -----------------------------
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            out = model(x)
            probs = torch.softmax(out, dim=1)

            y_true.append(y.item())
            y_pred.append(probs.argmax(dim=1).item())
            y_probs.append(probs.cpu().numpy()[0])

    y_probs = np.array(y_probs)

    # -----------------------------
    # METRICS
    # -----------------------------
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    rec = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    all_metrics.append([acc, prec, rec, f1])

    # -----------------------------
    # CONFUSION MATRIX
    # -----------------------------
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title(f"Confusion Matrix - Fold {fold+1}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # -----------------------------
    # ROC CURVES (One-vs-Rest)
    # -----------------------------
    plt.figure(figsize=(6, 5))
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(
            np.array(y_true) == i,
            y_probs[:, i]
        )
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr, tpr,
            label=f"{CLASS_NAMES[i]} (AUC={roc_auc:.2f})"
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Fold {fold+1}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, "roc_curve.png"), dpi=300)
    plt.close()

# =========================================================
# FINAL RESULTS (ALL FOLDS)
# =========================================================
metrics_df = pd.DataFrame(
    all_metrics,
    columns=["Accuracy", "Precision", "Recall", "F1"]
)

metrics_df.to_csv(
    os.path.join(RESULTS_DIR, "summary.csv"),
    index=False
)

mean_metrics = metrics_df.mean()
std_metrics = metrics_df.std()

with open(os.path.join(RESULTS_DIR, "mean_metrics.txt"), "w") as f:
    f.write("Mean ± Std (5-Fold Cross-Validation)\n\n")
    for m in metrics_df.columns:
        f.write(
            f"{m}: {mean_metrics[m]:.4f} ± {std_metrics[m]:.4f}\n"
        )

print("\n===== FINAL RESULTS =====")
print(metrics_df)
print("\nMean ± Std")
print(mean_metrics)
print(std_metrics)
