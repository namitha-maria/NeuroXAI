import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
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
NUM_CLASSES = 3
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4

# ----------------------------
# DATA
# ----------------------------
dataset = OASISDataset("neuroxai_data")

train_size = int(0.85 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size]
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"[INFO] Loaded {len(dataset)} subjects")
print(f"[INFO] Test samples: {len(test_dataset)}")

# ----------------------------
# MODEL
# ----------------------------
model = DenseNet_ViT(num_classes=NUM_CLASSES).to(device)

# ----------------------------
# LOSS (CLASS WEIGHTED)
# ----------------------------
class_weights = torch.tensor([1.0, 2.5, 4.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

# ----------------------------
# TRAINING
# ----------------------------
best_bal_acc = 0.0

for epoch in range(EPOCHS):
    # Unfreeze transformer after 10 epochs
    if epoch == 10:
        print("🔓 Unfreezing Transformer")
        for p in model.transformer.parameters():
            p.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5
    )

    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # ----------------------------
    # EVALUATION
    # ----------------------------
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Loss: {avg_loss:.4f} | "
        f"Acc: {acc:.4f} | "
        f"Bal Acc: {bal_acc:.4f}"
    )

    # Save best model (BALANCED accuracy)
    if bal_acc > best_bal_acc:
        best_bal_acc = bal_acc
        torch.save(model.state_dict(), "best_densenet_vit_oasis.pth")

# ----------------------------
# FINAL EVALUATION
# ----------------------------
print("\n================ TEST RESULTS ================")
print("Accuracy:", acc)
print("Balanced Accuracy:", bal_acc)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Normal", "Mild", "Moderate"]))

print("==============================================")
