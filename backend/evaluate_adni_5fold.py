import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.preprocessing import label_binarize

from backend.model import DenseNet_ViT
from backend.adni_dataset import ADNIDataset

# ---------------- CONFIG ----------------
DEVICE = torch.device("cpu")
MODEL_PATH = "best_densenet_vit_adni.pth"
DATA_ROOT = "neuroxai_data/ADNI_SLICES"

CLASS_NAMES = ["Normal", "Mild", "Moderate"]
NUM_CLASSES = 3
N_SPLITS = 5
BATCH_SIZE = 8

SAVE_DIR = "results/adni/5fold_eval"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- DATA ----------------
dataset = ADNIDataset(root=DATA_ROOT, slices=16)
labels = np.array([int(dataset[i][1]) for i in range(len(dataset))])

print(f"[INFO] Loaded {len(dataset)} ADNI subjects")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# ---------------- LOOP ----------------
for fold, (_, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    print(f"\n===== Fold {fold+1}/{N_SPLITS} =====")

    test_ds = torch.utils.data.Subset(dataset, test_idx)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = DenseNet_ViT(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            probs = torch.softmax(out, dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(out.argmax(1).cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # ---------- Confusion Matrix ----------
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5,5))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix – Fold {fold+1}")
    plt.xticks(range(NUM_CLASSES), CLASS_NAMES)
    plt.yticks(range(NUM_CLASSES), CLASS_NAMES)

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(j, i, cm[i,j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/confusion_matrix_fold_{fold+1}.png", dpi=300)
    plt.close()

    # ---------- ROC Curves ----------
    y_true_bin = label_binarize(y_true, classes=[0,1,2])

    plt.figure(figsize=(6,5))
    for c in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:,c], y_prob[:,c])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{CLASS_NAMES[c]} (AUC={roc_auc:.2f})")

    plt.plot([0,1], [0,1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – Fold {fold+1}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/roc_curve_fold_{fold+1}.png", dpi=300)
    plt.close()

print("\n✅ 5-fold evaluation complete")
