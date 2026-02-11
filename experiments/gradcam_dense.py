# training/gradcam_dense.py

import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAMDense:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, slice_tensor, class_idx=None):
        """
        slice_tensor: (1, 3, 224, 224)
        """

        self.model.zero_grad()

        # ---- CNN FORWARD ONLY ----
        feats = self.model.cnn(slice_tensor)          # (1, 1024)
        feats = self.model.proj(feats)                # (1, 256)
        logits = self.model.classifier(feats)         # (1, num_classes)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        score = logits[:, class_idx]
        score.backward()

        # ---- GRAD-CAM ----
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam
