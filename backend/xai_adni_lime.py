import os
import numpy as np
import torch
import cv2

from lime import lime_image
from skimage.segmentation import mark_boundaries

from backend.model import DenseNet_ViT
from backend.adni_dataset import ADNIDataset

# -----------------------
# CONFIG
# -----------------------
DEVICE = torch.device("cpu")
SAVE_DIR = "results/adni/lime"
os.makedirs(SAVE_DIR, exist_ok=True)

CLASS_NAMES = ["Normal", "Mild", "Moderate"]
NUM_CLASSES = 3
NUM_SLICES = 16

MODEL_PATH = "best_densenet_vit_adni.pth"
DATA_ROOT = "neuroxai_data/ADNI_SLICES"

# -----------------------
# LOAD MODEL
# -----------------------
model = DenseNet_ViT(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)
model.eval()

print("✅ ADNI model loaded")

# -----------------------
# LOAD DATASET
# -----------------------
dataset = ADNIDataset(
    root=DATA_ROOT,
    slices=NUM_SLICES
)

# -----------------------
# FIND ONE SUBJECT PER CLASS
# -----------------------
selected = {0: None, 1: None, 2: None}

for i in range(len(dataset)):
    _, y = dataset[i]
    label = int(y)

    if selected[label] is None:
        selected[label] = i

    if all(v is not None for v in selected.values()):
        break

print("[INFO] Selected subjects:", selected)

# -----------------------
# PREDICTION FUNCTION FOR LIME
# -----------------------
def predict_fn(images):
    """
    images: (N, H, W, 3) in [0,1]
    returns: (N, num_classes)
    """
    images = torch.tensor(images).permute(0, 3, 1, 2).float()
    images = images.unsqueeze(1)                 # (N, 1, 3, H, W)
    images = images.repeat(1, NUM_SLICES, 1, 1, 1)

    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

    return probs.cpu().numpy()

# -----------------------
# LIME EXPLAINER
# -----------------------
explainer = lime_image.LimeImageExplainer()

# -----------------------
# RUN LIME FOR EACH CLASS
# -----------------------
for label, idx in selected.items():
    x, y = dataset[idx]

    x = x.unsqueeze(0).to(DEVICE)

    # middle slice
    slice_idx = x.shape[1] // 2
    slice_tensor = x[:, slice_idx]               # (1, 3, 224, 224)

    image = slice_tensor[0].permute(1, 2, 0).cpu().numpy()
    image = np.clip(image, 0, 1)

    explanation = explainer.explain_instance(
        image=image,
        classifier_fn=predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    pred_class = explanation.top_labels[0]

    temp, mask = explanation.get_image_and_mask(
        label=pred_class,
        positive_only=True,
        num_features=5,
        hide_rest=True
    )

    # -----------------------
    # SAVE RESULTS
    # -----------------------
    class_name = CLASS_NAMES[label]

    orig = (image * 255).astype(np.uint8)
    top_features = (temp * 255).astype(np.uint8)

    overlay = mark_boundaries(orig, mask)
    overlay = (overlay * 255).astype(np.uint8)

    cv2.imwrite(f"{SAVE_DIR}/{class_name}_original.png", orig)
    cv2.imwrite(f"{SAVE_DIR}/{class_name}_top_features.png", top_features)
    cv2.imwrite(f"{SAVE_DIR}/{class_name}_overlay.png", overlay)

    print(f"✅ LIME generated for {class_name}")
    print(f"   True label: {CLASS_NAMES[y]}")
    print(f"   Predicted: {CLASS_NAMES[pred_class]}")

print("\n🎉 ADNI LIME generation complete")
