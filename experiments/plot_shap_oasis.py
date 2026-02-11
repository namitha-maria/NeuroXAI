# training/plot_shap_oasis.py

import matplotlib
matplotlib.use("Agg")  # no GUI

import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# CONFIG
# -----------------------
SHAP_DIR = "results/oasis/shap"
OUT_DIR = "results/oasis/shap_plots"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = ["Normal", "Mild", "Moderate"]

# -----------------------
# PLOT SHAP (CLASS LEVEL)
# -----------------------
for cls in CLASS_NAMES:
    shap_path = os.path.join(SHAP_DIR, f"shap_{cls}.npy")

    if not os.path.exists(shap_path):
        print(f"⚠️ Missing SHAP file for {cls}")
        continue

    shap_values = np.load(shap_path)[0]  # shape: (3,)

    plt.figure(figsize=(6, 4))
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in shap_values]

    plt.barh(CLASS_NAMES, shap_values, color=colors)
    plt.axvline(0, color="black", linewidth=0.8)

    plt.xlabel("SHAP value")
    plt.title(f"Class-level SHAP Explanation – True class: {cls}")
    plt.tight_layout()

    save_path = os.path.join(OUT_DIR, f"shap_{cls}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ SHAP plot saved: {save_path}")
