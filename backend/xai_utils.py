import torch
import numpy as np

def predict_with_fixed_slices(single_slice, fixed_slices, model, device):
    """
    single_slice: (H, W, 3) numpy array in [0,1]
    fixed_slices: (S, 3, H, W) numpy array
    """
    x = fixed_slices.copy()

    # explain middle slice
    slice_idx = fixed_slices.shape[0] // 2
    x[slice_idx] = single_slice.transpose(2, 0, 1)

    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)

    return probs.cpu().numpy()[0]
