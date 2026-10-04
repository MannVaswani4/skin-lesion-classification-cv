# Skin Lesion Classification — CNN + BiLSTM Attention Reproduction

> **Disclaimer**: This repository is an independent reproduction of the methodology described in the selected research paper. It is not the original authors' repository.

---

## 1. Project Overview
This repository faithfully reproduces the baseline methodology and benchmark results of the research paper:

> **"An Attention Based Hybrid Approach Using CNN and BiLSTM for Improved Skin Lesion Classification"**  
> *Ayesha Shaik, Shivanya Shomir Dutta, Ishaan Milind Sawant, Shreyas Kumar, Ananthakrishnan Balasundaram, Kanjar De*  
> Published in *Scientific Reports* (2025) 15:15680.  
> DOI: [10.1038/s41598-025-00025-2](https://doi.org/10.1038/s41598-025-00025-2)

The immediate objective of this project is strict scientific reproduction without introducing architectural modifications, unproven heuristics, or hyperparameter changes.

---

## 2. Research Paper & Dataset Reference

- **Paper PDF**: `docs/Research Paper CV.pdf`
- **Dataset**: `shawon250/ham10000-balanced-128x128` (Kaggle dataset derived from HAM10000)
- **Local Dataset Directory**: `data/ISIC_2018_128X128_6705/`
- **Total Dataset Size**: 46,935 RGB JPEG images ($128 \times 128$ resolution)

---

## 3. Dataset Structure & Class Distribution

The dataset is partitioned into seven distinct skin lesion classes across pre-defined `train`, `validation`, and `test` splits:

| Class | Description | Train Count | Val Count | Test Count | Total Count |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **AKIEC** | Actinic keratoses / Bowen's disease | 4,693 | 671 | 1,341 | 6,705 |
| **BCC** | Basal cell carcinoma | 4,693 | 671 | 1,341 | 6,705 |
| **BKL** | Benign keratosis-like lesions | 4,693 | 671 | 1,341 | 6,705 |
| **DF** | Dermatofibroma | 4,693 | 671 | 1,341 | 6,705 |
| **MEL** | Melanoma | 4,693 | 671 | 1,341 | 6,705 |
| **NV** | Melanocytic nevi | 4,693 | 671 | 1,341 | 6,705 |
| **VASC** | Vascular lesions | 4,693 | 671 | 1,341 | 6,705 |
| **TOTAL** | **All 7 Classes** | **32,851** | **4,697** | **9,387** | **46,935** |

---

## 4. Architecture Specifications

The paper proposed a hybrid architecture combining Convolutional Neural Networks (CNN), Bidirectional Long Short-Term Memory (BiLSTM) units, and multi-dimensional attention (Spatial, Channel, and Temporal Attention):

```
Input (224 × 224 × 3)
   │
   ▼
Conv2D (32, 3×3, ReLU, Same) ──► MaxPool2D (2×2)
   │
   ▼
Conv2D (64, 3×3, ReLU, Same) ──► MaxPool2D (2×2)
   │
   ▼
Conv2D (128, 3×3, ReLU, Same) ──► MaxPool2D (2×2)
   │
   ▼
Conv2D (256, 3×3, ReLU, Same) ──► MaxPool2D (2×2)  [Feature shape: 14 × 14 × 256]
   │
   ▼
Spatial Attention (Kernel: 7×7, Conv Sigmoid)
   │
   ▼
Channel Attention (Global Avg/Max Pool + Shared MLP)
   │
   ▼
Reshape (196 time steps, 256 feature dimensions)
   │
   ▼
BiLSTM Layer 1 (128 units, return_sequences=True)
   │
   ▼
BiLSTM Layer 2 (64 units, return_sequences=True)
   │
   ▼
Temporal Attention (tanh hidden projection + softmax context weighting)
   │
   ▼
Dense (512 units, ReLU) ──► Dropout (0.5) ──► Dense (7 units, Softmax)
```

---

## 5. Preprocessing & Augmentation Pipeline

- **Runtime Preprocessing**: Source $128 \times 128$ images are resized at runtime to $224 \times 224$ and pixel values rescaled by $1/255$.
- **Training Augmentation** (Table 4 parameters via `ImageDataGenerator`):
  - `rescale = 1/255`
  - `rotation_range = 20`
  - `width_shift_range = 0.1`
  - `height_shift_range = 0.1`
  - `zoom_range = 0.1`
  - `shear_range = 10`
  - `horizontal_flip = True`
  - `fill_mode = "nearest"`
- **Validation & Test Preprocessing**: Rescale $1/255$ only (no training augmentations).

---

## 6. Training Protocol

- **Optimizer**: Adam ($\text{lr} = 0.001$)
- **Loss Function**: Categorical Cross-Entropy (`categorical_crossentropy`)
- **Batch Size**: 32
- **Epochs**: 20
- **Callbacks**:
  - `ReduceLROnPlateau`: factor = 0.2, patience = 5, min_lr = 0.001
  - `EarlyStopping`: patience = 10, restore_best_weights = True
- **Random Seed**: 42

---

## 7. Installation & Requirements

1. Clone repository and install dependencies:
```bash
pip install -r requirements.txt
```

2. Run unit tests to verify dataset integrity and code modules:
```bash
python -m unittest discover -s tests
```

---

## 8. How to Run the Baseline Experiment

To train the baseline model (pending protocol review and approval):

```bash
python src/train.py --config configs/baseline.yaml
```

To run a dry-run check without training execution:

```bash
python src/train.py --config configs/baseline.yaml --dry-run
```

To evaluate a trained model on the test set:

```bash
python src/evaluate.py --config configs/baseline.yaml --model-path experiments/baseline/best_model.h5
```

---

## 9. Artifact Locations & Saved Results

- Config: `configs/baseline.yaml`
- Model Checkpoints: `experiments/baseline/best_model.h5`
- Logs & History: `experiments/baseline/history.json`
- Test Set Evaluation Results: `results/baseline/test_metrics.json`
- Confusion Matrix: `results/baseline/confusion_matrix.npy`

---

## 10. Reported Paper Results vs Reproduction Benchmark

| Metric | Paper Reported Result | Reproduction Result | Difference |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 92.73% | Pending Execution | TBD |
| **Precision** | 92.84% | Pending Execution | TBD |
| **F1 Score** | 92.70% | Pending Execution | TBD |
| **Recall / Sensitivity** | 92.73% | Pending Execution | TBD |
| **Jaccard Index (JAC)** | 87.08% | Pending Execution | TBD |
| **Dice Coefficient (DIC)** | 92.70% | Pending Execution | TBD |
| **MCC** | 91.55% | Pending Execution | TBD |
| **Specificity** | 98.79% | Pending Execution | TBD |
| **AUC** | 99.42% | Pending Execution | TBD |

---

## 11. Data Leakage Audit Findings & Known Limitations

As documented in [`docs/DATA_LEAKAGE_AUDIT.md`](docs/DATA_LEAKAGE_AUDIT.md), the dataset `shawon250/ham10000-balanced-128x128` was created by generating offline augmentations of the 10,015 HAM10000 source images to achieve 6,705 images per class, and then applying an image-level split. As a result, 2,722 original source lesion IDs present in `train` also have augmented variants in `test`.

Per reproduction protocol rules, **we do not alter or re-split the supplied dataset**, ensuring our evaluation strictly adheres to the exact dataset state used in the published paper.
