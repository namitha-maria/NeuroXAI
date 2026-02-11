import os
import nibabel as nib
import numpy as np
import cv2

# INPUT: original ADNI data (with .nii files)
ADNI_ROOT = r"C:\Users\Namitha Maria Joseph\neuroxai\neuroxai_data\ADNI"

# OUTPUT: axial PNG slices
OUT_ROOT = r"C:\Users\Namitha Maria Joseph\neuroxai\neuroxai_data\ADNI_SLICES"

CLASSES = ["Normal", "Mild", "Moderate"]
IMG_SIZE = 224


def normalize(img):
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return (img * 255).astype(np.uint8)


for cls in CLASSES:
    input_class = os.path.join(ADNI_ROOT, cls)
    output_class = os.path.join(OUT_ROOT, cls)

    if not os.path.exists(input_class):
        print(f"Skipping missing folder: {input_class}")
        continue

    os.makedirs(output_class, exist_ok=True)

    for subject in os.listdir(input_class):
        subject_path = os.path.join(input_class, subject)

        # ---- find .nii file (deep search) ----
        nii_path = None
        for root, _, files in os.walk(subject_path):
            for f in files:
                if f.endswith(".nii"):
                    nii_path = os.path.join(root, f)
                    break
            if nii_path is not None:
                break

        if nii_path is None:
            print(f"No .nii found for {subject}")
            continue

        print(f"Converting {cls} / {subject}")

        # ---- load NIfTI & FORCE CANONICAL ORIENTATION ----
        nii = nib.load(nii_path)
        nii = nib.as_closest_canonical(nii)   # 🔑 THIS LINE FIXES EVERYTHING
        volume = nii.get_fdata()

        out_subject = os.path.join(output_class, subject)
        os.makedirs(out_subject, exist_ok=True)

        # ---- extract TRUE axial slices (Z-axis) ----
        for i in range(volume.shape[2]):
            slice_img = volume[:, :, i]

            # skip empty / useless slices
            if np.std(slice_img) < 5:
                continue

            slice_img = normalize(slice_img)
            slice_img = cv2.resize(slice_img, (IMG_SIZE, IMG_SIZE))

            out_path = os.path.join(out_subject, f"slice_{i:03d}.png")
            cv2.imwrite(out_path, slice_img)
