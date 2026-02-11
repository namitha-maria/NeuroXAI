import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

CLASS_MAP = {"Normal": 0, "Mild": 1, "Moderate": 2}


class OASISDataset(Dataset):
    def __init__(self, root="neuroxai_data", slices=16):
        self.samples = []
        self.slices = slices

        for label, idx in CLASS_MAP.items():
            label_dir = os.path.join(root, label)
            if not os.path.exists(label_dir):
                continue

            for subject in os.listdir(label_dir):
                self.samples.append((os.path.join(label_dir, subject), idx))

        print(f"[INFO] Loaded {len(self.samples)} subjects")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        imgs = []

        slice_files = sorted(os.listdir(path))
        total = len(slice_files)

        # Center slice selection
        if total >= self.slices:
            start = (total - self.slices) // 2
            slice_files = slice_files[start:start + self.slices]
        else:
            slice_files = slice_files + [slice_files[-1]] * (self.slices - total)

        for f in slice_files:
            img = cv2.imread(os.path.join(path, f), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            img = np.stack([img, img, img], axis=0)
            imgs.append(img)

        imgs = torch.tensor(np.array(imgs), dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return imgs, label
