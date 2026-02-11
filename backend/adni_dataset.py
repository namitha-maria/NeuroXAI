import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

# Must match folder names exactly
CLASS_MAP = {
    "Normal": 0,
    "Mild": 1,
    "Moderate": 2
}


class ADNIDataset(Dataset):
    def __init__(self, root="neuroxai_data/ADNI_SLICES", slices=16):
        self.samples = []
        self.slices = slices

        for label_name, label_idx in CLASS_MAP.items():
            label_dir = os.path.join(root, label_name)
            if not os.path.exists(label_dir):
                continue

            for subject in os.listdir(label_dir):
                subject_path = os.path.join(label_dir, subject)
                if os.path.isdir(subject_path):
                    self.samples.append((subject_path, label_idx))

        print(f"[INFO] Loaded {len(self.samples)} ADNI subjects")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subject_path, label = self.samples[idx]

        slice_files = sorted(os.listdir(subject_path))
        total_slices = len(slice_files)

        # center slice selection
        if total_slices >= self.slices:
            start = (total_slices - self.slices) // 2
            slice_files = slice_files[start:start + self.slices]
        else:
            slice_files = slice_files + [slice_files[-1]] * (self.slices - total_slices)

        imgs = []

        for f in slice_files:
            img = cv2.imread(os.path.join(subject_path, f), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            img = np.stack([img, img, img], axis=0)
            imgs.append(img)

        imgs = torch.tensor(np.array(imgs), dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return imgs, label
