# NeuroXAI

**AI-Powered Dementia Stage Analysis with Explainable Insights**

NeuroXAI is an explainable AI system for dementia stage classification from brain MRI scans. It combines deep learning with interpretability techniques to provide transparent, clinically-relevant predictions.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Explainability](#explainability)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

NeuroXAI addresses the critical need for interpretable AI in medical diagnostics by providing:

- **Subject-level predictions** for dementia stage classification (Normal, Mild, Moderate)
- **Explainability visualizations** using Grad-CAM to highlight influential brain regions
- **User-friendly web interface** for clinicians and researchers
- **FastAPI backend** for scalable deployment

The system is trained on OASIS and ADNI datasets and achieves competitive performance while maintaining transparency through explainable AI techniques.

---

## Features

### Core Capabilities
- **Multi-stage Classification**: Classify dementia into Normal, Mild, or Moderate stages
- **Explainable Predictions**: Grad-CAM visualizations show which brain regions influenced the AI's decision
- **NIfTI Support**: Direct upload and processing of `.nii` and `.nii.gz` MRI files
- **Cross-validation**: 5-fold cross-validation for robust performance evaluation
- **Interactive UI**: Modern, responsive web interface

### Technical Features
- Hybrid DenseNet-Vision Transformer architecture
- Subject-level aggregation across multiple MRI slices
- Canonical orientation standardization (RAS)
- Class-weighted loss for imbalanced datasets
- CPU and GPU support

---

## Architecture

### Model: DenseNet-ViT Hybrid

```
Input MRI (.nii) → Slice Extraction → CNN Feature Extraction → Transformer Encoding → Classification
                    (16 slices)      (DenseNet121)           (ViT Encoder)       (3 classes)
```

**Key Components:**
1. **DenseNet121 Backbone**: Pretrained CNN for spatial feature extraction from individual slices
2. **Linear Projection**: Maps CNN features to transformer embedding space (256-dim)
3. **Positional Embeddings**: Captures slice ordering information
4. **Transformer Encoder**: Aggregates features across slices using self-attention
5. **Classification Head**: Final prediction with confidence scores

**Training Strategy:**
- Stage 1 (Epochs 1-10): Freeze transformer, fine-tune DenseNet's final layers
- Stage 2 (Epochs 11+): Unfreeze transformer for end-to-end training

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for faster training

### Clone the Repository
```bash
git clone https://github.com/namitha-maria/neuroxai.git
cd neuroxai
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

**Core Dependencies:**
```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
fastapi>=0.100.0
uvicorn>=0.23.0
nibabel>=5.0.0
opencv-python>=4.8.0
scikit-learn>=1.3.0
numpy>=1.24.0
matplotlib>=3.7.0
pillow>=10.0.0
einops>=0.6.0
lime>=0.2.0
shap>=0.42.0
```

### Download Pre-trained Models
```bash
# Place the trained model checkpoint in the project root
# best_densenet_vit_adni.pth (for ADNI dataset)
# best_densenet_vit_oasis.pth (for OASIS dataset)
```

---

## Usage

### 1. Quick Start - Web Interface

**Start the Backend API:**
```bash
cd backend
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Open the Frontend:**
```bash
cd frontend
# Open index.html in your browser
# Or serve with a simple HTTP server:
python -m http.server 8080
```

Navigate to `http://localhost:8080` and upload a `.nii` MRI file.

### 2. Command-Line Prediction

**Single Subject Prediction:**
```python
from backend.model import DenseNet_ViT
import torch
import nibabel as nib

# Load model
model = DenseNet_ViT(num_classes=3)
model.load_state_dict(torch.load("best_densenet_vit_adni.pth"))
model.eval()

# Load and preprocess MRI
# [Add preprocessing code here]

# Predict
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1)
    print(f"Predicted Class: {['Normal', 'Mild', 'Moderate'][prediction]}")
```

### 3. Training from Scratch

**Prepare Your Dataset:**
```
neuroxai_data/
├── Normal/
│   ├── subject_001/
│   │   ├── slice_000.png
│   │   ├── slice_001.png
│   │   └── ...
├── Mild/
└── Moderate/
```

**Train the Model:**
```bash
python backend/train.py
```

**Evaluate Performance:**
```bash
python backend/test_model.py
```

**Run Cross-Validation:**
```bash
python experiments/crossval.py
```

---

## Dataset

### Supported Datasets

#### OASIS (Open Access Series of Imaging Studies)
- **Classes**: Normal, Mild, Moderate
- **Format**: Axial T1-weighted MRI slices (PNG)
- **Preprocessing**: Center-cropped 16 slices per subject

#### ADNI (Alzheimer's Disease Neuroimaging Initiative)
- **Classes**: Normal, Mild, Moderate  
- **Format**: NIfTI (`.nii`, `.nii.gz`)
- **Preprocessing**: 
  - Canonical orientation standardization (RAS)
  - Axial slice extraction
  - Normalization to [0, 1]

### Data Preprocessing Pipeline

```python
# Convert ADNI NIfTI to slices
python backend/convert_adni_to_slices.py
```

**Steps:**
1. Load NIfTI volume
2. Apply canonical orientation (`nib.as_closest_canonical`)
3. Extract 16 center axial slices
4. Resize to 224×224
5. Normalize intensity values
6. Convert to RGB (3-channel)

---

## Model Performance

### ADNI Dataset Results

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 85.2% |
| **Balanced Accuracy** | 82.7% |
| **ROC-AUC (OvR)** | 0.8036 |

**Class-wise Performance:**
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Normal | 0.92 | 0.88 | 0.90 |
| Mild | 0.78 | 0.81 | 0.79 |
| Moderate | 0.83 | 0.79 | 0.81 |

### OASIS Dataset Results

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 55.6% |
| **Balanced Accuracy** | 51.1% |

*Note: Lower performance on OASIS likely due to smaller dataset size and class imbalance.*

### Cross-Validation (5-Fold)
- **Mean Accuracy**: 78.3% ± 4.2%
- **Mean F1-Score**: 0.76 ± 0.05

---

## Project Structure

```
neuroxai/
├── backend/
│   ├── api.py                    # FastAPI endpoints
│   ├── model.py                  # DenseNet-ViT architecture
│   ├── dataset.py                # OASIS dataset loader
│   ├── adni_dataset.py           # ADNI dataset loader
│   ├── train.py                  # Training script
│   ├── test_model.py             # Evaluation script
│   ├── test_adni.py              # ADNI-specific testing
│   ├── gradcam_utils.py          # Grad-CAM implementation
│   ├── xai_adni_gradcam.py       # XAI visualization generation
│   └── convert_adni_to_slices.py # NIfTI preprocessing
│
├── frontend/
│   ├── index.html                # Main web interface
│   ├── landing.html              # Alternative landing page
│   ├── style.css                 # Styling
│   └── assets/                   # Images, icons
│
├── experiments/
│   ├── crossval.py               # 5-fold cross-validation
│   ├── generate_lime_oasis.py    # LIME explainability
│   └── generate_shap_oasis.py    # SHAP explainability
│
├── results/
│   ├── oasis/                    # OASIS experiment results
│   ├── adni/                     # ADNI experiment results
│   └── xai/                      # Explainability visualizations
│
├── neuroxai_data/                # Dataset directory
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## API Reference

### Endpoints

#### `POST /predict_nii`
Predict dementia stage from NIfTI file.

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/predict_nii" \
  -F "file=@scan.nii.gz"
```

**Response:**
```json
{
  "stage": "Mild",
  "confidence": 87.43
}
```

#### `POST /gradcam`
Generate Grad-CAM explainability visualizations.

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/gradcam" \
  -F "file=@scan.nii.gz"
```

**Response:**
```json
{
  "original_slices": ["base64_image_1", "base64_image_2", ...],
  "overlay_slices": ["base64_overlay_1", "base64_overlay_2", ...]
}
```

#### `GET /`
Health check endpoint.

**Response:**
```json
{
  "status": "NeuroXAI ADNI backend running"
}
```

---

## Explainability

NeuroXAI implements multiple XAI techniques:

### 1. Grad-CAM (Gradient-weighted Class Activation Mapping)
- Highlights brain regions that most influenced the model's prediction
- Applied to the final DenseNet convolutional layer
- Provides slice-by-slice heatmaps

**Example:**
```python
from backend.gradcam_utils import GradCAM

gradcam = GradCAM(model=model, target_layer=model.cnn.features.denseblock4)
cam = gradcam.generate(input_tensor, predicted_class)
```

### 2. LIME (Local Interpretable Model-agnostic Explanations)
- Explains individual predictions using superpixel perturbations
- Available in `experiments/generate_lime_oasis.py`

### 3. SHAP (SHapley Additive exPlanations)
- Game-theoretic approach to feature importance
- Available in `experiments/generate_shap_oasis.py`

---

## Contributing

Contributions are welcome! Please follow these guidelines:

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Check code style
flake8 backend/ experiments/
```

### Areas for Contribution
- [ ] Additional explainability methods (Integrated Gradients, Attention Rollout)
- [ ] Support for more datasets (AIBL, UK Biobank)
- [ ] Multi-modal fusion (MRI + PET + clinical data)
- [ ] Deployment optimizations (ONNX, TensorRT)
- [ ] Mobile app integration
- [ ] Clinical validation studies

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Disclaimer

**For Research and Educational Purposes Only**

This tool is **not intended** to diagnose, treat, cure, or prevent any disease. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any questions regarding a medical condition.

---

## Acknowledgments

- **OASIS**: Open Access Series of Imaging Studies (Washington University)
- **ADNI**: Alzheimer's Disease Neuroimaging Initiative
- **PyTorch**: Deep learning framework
- **timm**: PyTorch Image Models library
- **FastAPI**: Modern web framework for building APIs
- **Grad-CAM**: Original implementation by Selvaraju et al.

---

## Contact

For questions, issues, or collaboration opportunities:

- **GitHub Issues**: [Create an issue](https://github.com/namitha-maria/neuroxai/issues)
- **Email**: namithamj04@gmail.com
- **Project Website**: [neuroxai-project.github.io](https://neuroxai-project.github.io)

---

## Roadmap

### Version 1.0 (Current)
- [x] DenseNet-ViT hybrid architecture
- [x] OASIS and ADNI dataset support
- [x] Grad-CAM explainability
- [x] Web interface
- [x] FastAPI backend

### Version 2.0 (Planned)
- [ ] Real-time predictions with streaming
- [ ] Multi-site validation
- [ ] Uncertainty quantification
- [ ] Longitudinal tracking dashboard
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP)

### Future Directions
- [ ] 3D convolutional networks
- [ ] Transformer-only architecture
- [ ] Multi-task learning (age, gender, APOE genotype)
- [ ] Federated learning support
- [ ] Integration with PACS systems

---

## Citation

If you use NeuroXAI in your research, please cite:

```bibtex
@software{neuroxai2024,
  title={NeuroXAI: Explainable AI for Dementia Stage Classification},
  author={Namitha Maria Joseph},
  year={2026},
  url={https://github.com/namitha-maria/neuroxai}
}
```

---

**Advancing Explainable AI in Healthcare**
