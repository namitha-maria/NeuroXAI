import os
import io
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import base64
import torch
import cv2
import numpy as np
import nibabel as nib

# gradcam
from .gradcam_utils import GradCAM
from .model import DenseNet_ViT

# -----------------------
# CONFIG
# -----------------------
device = "cpu"   # change to "cuda" if available
SLICE_COUNT = 16
CLASS_NAMES = ["Normal", "Mild", "Moderate"]  # ADNI labels

# -----------------------
# PATHS
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "best_densenet_vit_adni.pth")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = DenseNet_ViT(num_classes=3).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -----------------------------
# FASTAPI APP
# -----------------------------
app = FastAPI(title="NeuroXAI ADNI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# UTILS: NIFTI → SLICES
# -----------------------------
def nifti_to_slices(nii_path, num_slices=16):
    img = nib.load(nii_path)

    # 🔥 Standardize orientation to RAS
    img = nib.as_closest_canonical(img)

    volume = img.get_fdata()


    # Normalize
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

    total = volume.shape[2]

    # Center slice selection
    if total >= num_slices:
        start = (total - num_slices) // 2
        indices = range(start, start + num_slices)
    else:
        indices = list(range(total)) + [total - 1] * (num_slices - total)

    slices = []
    for i in indices:
        sl = volume[:, :, i]
        sl = cv2.resize(sl, (224, 224))
        sl = np.stack([sl, sl, sl], axis=0)
        slices.append(sl)

    return torch.tensor(np.array(slices), dtype=torch.float32)  # (S,3,224,224)

# -----------------------------
# PREDICTION (NIFTI)
# -----------------------------
@app.post("/predict_nii")
async def predict_nii(file: UploadFile = File(...)):
    if not file.filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(status_code=400, detail="Upload .nii or .nii.gz file")

    # Save temp NIfTI
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp:
        tmp.write(await file.read())
        nii_path = tmp.name

    slices = nifti_to_slices(nii_path, SLICE_COUNT)
    os.remove(nii_path)

    x = slices.unsqueeze(0).to(device)  # (1,S,3,224,224)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0]
        pred = probs.argmax().item()

    return {
        "stage": CLASS_NAMES[pred],
        "confidence": round(probs[pred].item() * 100, 2)
    }

# -----------------------------
# GRAD-CAM (NIFTI)
# -----------------------------
import base64

@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):
    if not file.filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(status_code=400, detail="Upload .nii or .nii.gz file")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp:
        tmp.write(await file.read())
        nii_path = tmp.name

    slices = nifti_to_slices(nii_path, SLICE_COUNT)
    os.remove(nii_path)

    x = slices.unsqueeze(0).to(device)

    outputs = model(x)
    pred_class = outputs.argmax(dim=1).item()

    cam_extractor = GradCAM(
        model=model,
        target_layer=model.cnn.features.denseblock4
    )

    cams = cam_extractor.generate(x, pred_class)

    original_images = []
    overlay_images = []

    for i in range(SLICE_COUNT):

        # Original slice
        slice_img = slices[i].numpy()[0]
        slice_img = (slice_img * 255).astype(np.uint8)
        slice_rgb = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2BGR)

        # CAM
        cam = cv2.resize(cams[i], (224, 224))
        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(slice_rgb, 0.6, heatmap, 0.4, 0)

        # Encode
        _, buf1 = cv2.imencode(".png", slice_rgb)
        _, buf2 = cv2.imencode(".png", overlay)

        original_images.append(base64.b64encode(buf1).decode("utf-8"))
        overlay_images.append(base64.b64encode(buf2).decode("utf-8"))

    return {
        "original_slices": original_images,
        "overlay_slices": overlay_images
    }

# -----------------------------
# ROOT CHECK
# -----------------------------
@app.get("/")
def root():
    return {"status": "NeuroXAI ADNI backend running"}
