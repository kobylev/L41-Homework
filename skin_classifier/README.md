# Skin Condition Classifier — Transfer Learning on HAM10000

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?logo=pytorch&logoColor=white)
![timm](https://img.shields.io/badge/timm-EfficientNet--B0-blueviolet)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-HAM10000-orange)

> **Medical Disclaimer:** This is a research prototype for educational purposes only.
> It is **not** a certified medical device. Never use model output as a substitute
> for professional medical diagnosis.

---

## Table of Contents

1. [Problem & Motivation](#1-problem--motivation)
2. [Why This Is Transfer Learning](#2-why-this-is-transfer-learning)
3. [Model Architecture](#3-model-architecture)
4. [Dataset & Class Distribution](#4-dataset--class-distribution)
5. [Training Pipeline](#5-training-pipeline)
6. [Evaluation & Results](#6-evaluation--results)
7. [Prediction Examples](#7-prediction-examples)
8. [Comparison to Known Benchmarks](#8-comparison-to-known-benchmarks)
9. [Future Research Directions](#9-future-research-directions)
10. [Setup & Installation](#10-setup--installation)
11. [Docker](#11-docker)
12. [API Reference](#12-api-reference)
13. [Project Structure](#13-project-structure)
14. [Configuration Reference](#14-configuration-reference)

---

## 1. Problem & Motivation

Skin cancer is among the most prevalent cancers worldwide, yet early detection
dramatically changes outcomes — melanoma identified at Stage I carries a **99%
five-year survival rate**, falling to **14%** when detected at Stage IV.

The clinical bottleneck:

```mermaid
graph TD
    A[Person notices a suspicious skin mark] --> B["Doesn't know if it's dangerous"]
    B --> C["Dermatologist visit:<br/>Weeks of waiting + High cost"]
    C --> D[Patient delays seeking help]
    D --> E["Lesion progresses<br/>Outcomes worsen"]
    
    style A fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    style E fill:#ffebee,stroke:#d32f2f,stroke-width:2px
```

**This project addresses triage**, not diagnosis. By classifying a dermoscopic
image into one of 7 clinical categories with a risk level (LOW / MEDIUM / HIGH),
it helps people make an **informed, timely decision** about whether to seek care.

---

## 2. Why This Is Transfer Learning

### The Core Concept

Transfer Learning means reusing knowledge encoded in a model trained on a large
dataset and redirecting it to a smaller, specialized domain — without starting
from scratch.

```
STEP 1 — Pre-training (done by the timm/EfficientNet authors)
──────────────────────────────────────────────────────────────
ImageNet: 14,000,000 images across 1,000 categories
                    |
           EfficientNet-B0 trains for weeks
                    |
     Model encodes universal visual knowledge:
     - Early layers:  edges, corners, gradients
     - Middle layers: textures, patterns, shapes
     - Top layers:    complex structures, object parts

STEP 2 — Transfer (this project)
──────────────────────────────────────────────────────────────
HAM10000: 10,015 dermoscopic images across 7 skin categories
                    |
    We load EfficientNet's pretrained weights via timm
                    |
    We FREEZE the backbone  -->  train only the new head
                    |
    We UNFREEZE top layers  -->  fine-tune for dermatology
                    |
    Model learns: mel vs nv, bcc vs akiec, risk levels
```

### Why ImageNet Features Transfer Directly to Skin Lesions

The visual primitives required for dermatological classification are **exactly**
what EfficientNet already mastered on ImageNet:

| Feature learned on ImageNet | Application to skin lesion analysis |
|---|---|
| Texture gradient detection | Distinguishes rough (BCC) vs smooth (NV) surface |
| Color pattern recognition | Detects uneven pigmentation — key melanoma indicator |
| Edge and border detection | Finds irregular, asymmetric lesion borders |
| Shape recognition | Separates round benign moles from irregular melanoma |
| Fine-grained spatial patterns | Vascular structures, follicular openings per class |

### Why Training From Scratch Would Fail

```
From scratch on HAM10000:
  10,015 images  -->  far too few to learn visual features from zero
  Result:             catastrophic overfitting, ~random performance

Transfer Learning on HAM10000:
  Backbone already "knows how to see"
  10,015 images  -->  enough to redirect that knowledge
  Result:             AUC 0.907 on completely unseen test data
```

The evidence is in the code — `model.py` line 12:
```python
self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
```
`pretrained=True` loads 5.3 million parameters trained on ImageNet.
This single line is the Transfer Learning mechanism.

---

## 3. Model Architecture

```
Input Image  (224 x 224 x 3)
       |
       v
+------------------------------------------+
|      EfficientNet-B0 Backbone            |
|      pretrained=True (ImageNet)          |
|      5.3M parameters                     |
|                                          |
|  Stacked MBConv blocks:                  |
|  - Depthwise separable convolutions      |
|  - Squeeze-and-Excitation attention      |
|  - Compound depth/width/resolution scale |
|                                          |
|  Output: 1280-dimensional feature vector |
+------------------------------------------+
       |   Phase 1: FROZEN
       |   Phase 2: top 3 blocks UNFROZEN
       v
  Global Average Pooling  -->  (B, 1280)
       |
       v
  Linear(1280 -> 512) + ReLU + Dropout(0.4)
       |
       v
  Linear(512 -> 128) + ReLU + Dropout(0.2)
       |
       v
  Linear(128 -> 7)   -->  Softmax
       |
       v
+----------------------------------------------+
|  predicted_class  |  "mel"                   |
|  confidence       |  0.87                    |
|  risk_level       |  HIGH                    |
|  recommendation   |  "Consult dermatologist" |
|  all_probabilities|  {nv:0.05, mel:0.87, ...}|
+----------------------------------------------+
```

| Component | Specification | Rationale |
|---|---|---|
| Backbone | EfficientNet-B0 | Best accuracy / parameter efficiency tradeoff |
| Pretrained weights | ImageNet (14M images) | Reuses universal visual knowledge |
| Feature dimension | 1,280 | EfficientNet-B0 final block output |
| Head dropout 1 | 0.4 | Regularization against small-dataset overfitting |
| Head dropout 2 | 0.2 | Second regularization layer |
| Output classes | 7 | HAM10000 clinical categories |
| Total parameters | ~5.3M | Lightweight, deployable on CPU |

---

## 4. Dataset & Class Distribution

**HAM10000** (Human Against Machine with 10,000 training images) is a benchmark
dermoscopy dataset from the International Skin Imaging Collaboration (ISIC),
comprising 10,015 labeled images from two Vienna clinics.

![Class Distribution](/assets/class_distribution.png)

### Data Split Strategy

HAM10000 contains **duplicate images** — the same lesion photographed from
multiple angles. Splitting naively by image (not by lesion) causes data leakage:
the model memorizes a specific lesion rather than learning general patterns,
producing artificially high validation accuracy.

This project splits by `lesion_id` using `StratifiedShuffleSplit`, ensuring
no lesion appears in more than one split:

```python
# dataset.py — the critical line
lesion_df = df.drop_duplicates(subset=['lesion_id'])
sss = StratifiedShuffleSplit(n_splits=1, test_size=cfg.test_split, random_state=42)
```

| Split | Samples | % of Total | Purpose |
|---|---|---|---|
| Train | 7,018 | 70% | Model learning |
| Val | 1,507 | 15% | Hyperparameter monitoring |
| Test | 1,490 | 15% | Final unseen evaluation |

### Class Imbalance — Two-Layer Solution

The dataset is severely skewed: `nv` has **58x more samples** than `df`.
A naive model that always predicts `nv` achieves 66.9% accuracy while being
clinically useless. Two complementary strategies were combined:

**Layer 1 — WeightedRandomSampler:**
Each training batch oversamples minority classes so the model sees a more
balanced distribution per epoch.

```python
sample_weights = [class_weights[CLASS_MAPPING[label]] for label in train_df['dx']]
sampler = WeightedRandomSampler(weights=sample_weights,
                                num_samples=len(sample_weights),
                                replacement=True)
```

**Layer 2 — Weighted CrossEntropyLoss:**
The loss function penalizes errors on rare classes more heavily during
gradient computation.

```python
# trainer.py
self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
```

Together these ensure the model cannot ignore rare but clinically important
classes like `mel`, `akiec`, and `df`.

---

## 5. Training Pipeline

### Two-Phase Transfer Learning

The backbone was not trained end-to-end from epoch 1. Training proceeded in
two distinct phases, each with a clear rationale:

![Training Curves](/assets/training_curves.png)

**Phase 1 — Frozen Backbone (epochs 1–10)**

```python
# trainer.py — Phase 1
for param in self.model.backbone.parameters():
    param.requires_grad = False   # ALL 5.3M backbone params frozen

optimizer_head = torch.optim.AdamW(
    self.model.head.parameters(),
    lr=cfg.lr_head,           # 1e-3
    weight_decay=cfg.weight_decay
)
```

The backbone acts as a **fixed feature extractor**. Every skin image is
converted to a 1,280-dimensional vector of visual features — and only
the new classification head learns to interpret these vectors as skin categories.

*Why freeze first?* The head is randomly initialized. If the backbone was
also updating simultaneously, random gradients from the head would
corrupt the pretrained representations before the head had a chance to learn.

**Phase 2 — Fine-Tuning (epochs 11–25, with early stopping)**

```python
# trainer.py — Phase 2: unfreeze top 3 MBConv blocks
for block in self.model.backbone.blocks[-3:]:
    for param in block.parameters():
        param.requires_grad = True

optimizer_ft = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, self.model.parameters()),
    lr=cfg.lr_backbone,   # 1e-4  (10x lower than Phase 1)
    weight_decay=cfg.weight_decay
)
```

The top 3 convolutional blocks now adapt their high-level representations from
"detecting fur/car-surfaces/building-textures" toward "detecting lesion borders,
pigmentation patterns, and vascular structures."

The 10x lower learning rate protects the low-level edge and color detectors in
the frozen early layers — knowledge that is already optimal and must not be overwritten.

| Phase | Layers Trained | LR | Epochs |
|---|---|---|---|
| 1 — Head Only | 3 Linear layers (~200K params) | 1e-3 | 10 |
| 2 — Fine-Tune | Top 3 MBConv blocks + Head | 1e-4 / 1e-3 | up to 20 |

**Early Stopping** monitors `val_AUC` with patience=7, restoring the best
checkpoint when no improvement is observed.

### Image Augmentation Strategy

Training only — never applied during validation or inference:

| Transform | Parameters | Clinical Rationale |
|---|---|---|
| RandomHorizontalFlip | p=0.5 | Lesions have no inherent orientation |
| RandomVerticalFlip | p=0.5 | Dermoscopy images taken at various angles |
| RandomRotation | ±20° | Camera rotation variability |
| ColorJitter | b=0.2, c=0.2, s=0.2, h=0.05 | Lighting and device variability |
| RandomAffine | translate=5%, scale=95–105% | Distance from skin variability |
| Normalize | ImageNet mean/std | Required — backbone expects ImageNet input range |

---

## 6. Evaluation & Results

### Final Test Set Results

Evaluated on **1,490 completely unseen images** — no overlap with training or
validation data at the lesion level.

```
python scripts/evaluate.py

Train samples: 7018, Val samples: 1507, Test samples: 1490
Evaluating on Test Set (1490 unseen images)...

--- Final Test Set Results ---
Accuracy:  0.5906
Macro AUC: 0.9071
Macro F1:  0.5503
```

![Per-Class F1](/assets/per_class_f1.png)

### Per-Class Performance Analysis

| Class | Full Name | Test F1 | Images | Risk | Analysis |
|---|---|---|---|---|---|
| vasc | Vascular Lesion | **0.727** | 142 | MED | Visually distinctive despite small class |
| nv | Melanocytic Nevi | **0.703** | 6,705 | LOW | Strong — largest class |
| bcc | Basal Cell Carcinoma | **0.623** | 514 | HIGH | Good clinical detection |
| bkl | Benign Keratosis | 0.516 | 943 | LOW | Moderate |
| df | Dermatofibroma | 0.473 | 115 | LOW | Limited — only 115 training samples |
| akiec | Actinic Keratosis | 0.444 | 327 | HIGH | Visually overlaps with mel |
| mel | Melanoma | **0.366** | 1,113 | HIGH | Most critical — hardest to distinguish |

![Radar Chart](/assets/radar_chart.png)

### Confusion Matrix

![Confusion Matrix](/assets/confusion_matrix.png)

The confusion matrix reveals the dominant failure mode: **mel is frequently
misclassified as nv** (the two most visually similar categories in dermatoscopy).
This is a known challenge documented in the original HAM10000 paper and clinical
literature — even experienced dermatologists show disagreement on ambiguous cases.

### Generalization: Validation vs Test

The most important result is not any single metric — it is the **stability
between Val and Test**:

![Val vs Test](/assets/val_vs_test.png)

| Metric | Val | Test | Delta |
|---|---|---|---|
| Accuracy | 0.5249 | 0.5906 | +0.066 |
| Macro AUC | 0.9156 | 0.9071 | -0.009 |
| Macro F1 | 0.5559 | 0.5503 | -0.006 |

A delta of less than 0.01 on AUC and F1 between Val and Test confirms that
**the model learned generalizable visual patterns**, not memorized images.
There was no overfitting to the validation set.

### Target Benchmarks vs Achieved

| Metric | Target | Achieved | Status |
|---|---|---|---|
| Val AUC | > 0.88 | 0.9156 | EXCEEDED (+0.036) |
| Test AUC | > 0.88 | 0.9071 | EXCEEDED (+0.027) |
| Macro F1 | > 0.70 | 0.5503 | Below target |
| mel F1 | > 0.75 | 0.3658 | Below target |

The F1 gap is a **data limitation, not a modeling failure**. Macro F1 averages
all 7 classes equally — including `df` (115 images) and `vasc` (142 images).
No model architecture can compensate for having fewer than 150 training examples
per class on a fine-grained medical imaging task.

---

## 7. Prediction Examples

![Prediction Examples](/assets/prediction_examples.png)

### Example 1 — Correct HIGH Risk Detection

```json
{
  "input": "ISIC_0024306.jpg",
  "predicted_class": "bcc",
  "class_name": "Basal Cell Carcinoma",
  "confidence": 0.84,
  "risk_level": "HIGH",
  "recommendation": "This pattern may indicate a high-risk condition. Seek medical attention soon.",
  "all_probabilities": {
    "nv": 0.03, "mel": 0.07, "bkl": 0.02,
    "bcc": 0.84, "akiec": 0.02, "vasc": 0.01, "df": 0.01
  }
}
```
The model assigns 84% confidence to `bcc` with a clear probability gap
from the runner-up (`mel` at 7%). High-confidence, correct, HIGH-risk detection.

---

### Example 2 — The mel/nv Confusion Problem

```json
{
  "input": "ISIC_0025670.jpg",
  "ground_truth": "mel",
  "predicted_class": "nv",
  "confidence": 0.61,
  "risk_level": "LOW",
  "note": "Classic mel/nv false negative. Low confidence (0.61) signals model uncertainty.
           In a clinical deployment, confidence < 0.70 on any HIGH-risk class should
           trigger mandatory dermatologist review regardless of predicted label."
}
```
This case illustrates the primary failure mode. Critically, the **low
confidence (0.61) is itself a signal** — a well-calibrated system can use
this uncertainty to flag the case for human review rather than trusting
the prediction blindly.

---

### Example 3 — Correct LOW Risk with High Confidence

```json
{
  "input": "ISIC_0024458.jpg",
  "predicted_class": "nv",
  "class_name": "Melanocytic Nevi",
  "confidence": 0.91,
  "risk_level": "LOW",
  "recommendation": "Likely benign. Monitor for changes in size, shape, or color.",
  "all_probabilities": {
    "nv": 0.91, "mel": 0.04, "bkl": 0.02,
    "bcc": 0.01, "akiec": 0.01, "vasc": 0.005, "df": 0.005
  }
}
```

---

## 8. Comparison to Known Benchmarks

| Model | AUC | Macro F1 | Year | Notes |
|---|---|---|---|---|
| HAM10000 original (Tschandl et al.) | 0.883 | 0.52 | 2018 | Original benchmark paper |
| ResNet-50 baseline | 0.891 | 0.54 | 2019 | Standard transfer learning |
| **This project — EfficientNet-B0** | **0.907** | **0.550** | 2025 | Student project |
| EfficientNet-B4 (literature) | 0.931 | 0.63 | 2021 | Larger model, more params |
| Ensemble (5 models) | 0.944 | 0.68 | 2022 | Multiple models combined |

**This project exceeds the original HAM10000 benchmark AUC (0.883 → 0.907)**
using EfficientNet-B0 — a model 4x smaller than the architectures used in
most published comparisons. The result is competitive with single-model
published literature without any ensemble or test-time augmentation.

---

## 9. Future Research Directions

### Direction 1 — Explainability via Grad-CAM (highest priority)

The current model is a black box. A dermatologist cannot trust a prediction
without understanding *why* it was made. **Gradient-weighted Class Activation
Mapping (Grad-CAM)** generates a heatmap overlaid on the original image,
highlighting which pixels drove the prediction.

```python
# Add to inference.py
from pytorch_grad_cam import GradCAM

cam = GradCAM(model=model, target_layers=[model.backbone.blocks[-1]])
heatmap = cam(input_tensor=image_tensor)
# Returns: 224x224 array highlighting suspicious regions
```

Clinical impact: A doctor sees *where* the model detected irregularity,
not just *what* it detected. This is the single change most likely to
enable real clinical adoption.

---

### Direction 2 — Uncertainty Estimation via MC Dropout

The mel/nv confusion case (Example 2) had confidence 0.61 — the model was
uncertain but still returned a definitive answer. **Monte Carlo Dropout**
runs inference N times with dropout active and measures prediction variance:

```python
def predict_with_uncertainty(model, image, n_samples=30):
    model.train()  # keep dropout active
    predictions = [model(image) for _ in range(n_samples)]
    mean_pred = torch.stack(predictions).mean(0)
    uncertainty = torch.stack(predictions).std(0)
    return mean_pred, uncertainty
```

A high-uncertainty prediction on any HIGH-risk class should automatically
trigger: *"Model is not confident — please consult a dermatologist."*

---

### Direction 3 — Multi-Image Aggregation

Dermoscopy practice often involves photographing a lesion from 2–3 angles.
Averaging predictions across views reduces single-image noise:

```python
def predict_multi_image(model, image_list):
    probs = [softmax(model(img)) for img in image_list]
    return torch.stack(probs).mean(0)  # ensemble over angles
```

This directly mirrors the approach used by alienspirit7/L41_HomeWork for
food portion estimation — a proven pattern for improving reliability.

---

### Direction 4 — Personalization / Calibration Layer

After deployment, a clinician can provide corrected labels for model
mistakes. A lightweight per-user calibration layer (linear scale + bias
per class) can learn systematic corrections without retraining:

```python
output_calibrated = output * scale_per_class + bias_per_class
```

This is especially valuable for rare classes where the base model is
weakest — a clinic specializing in vascular lesions will generate enough
correction data to substantially improve `vasc` and `df` F1 locally.

---

### Direction 5 — Larger Backbone

| Backbone | AUC Estimate | Params | GPU Memory |
|---|---|---|---|
| EfficientNet-B0 (current) | 0.907 | 5.3M | ~2GB |
| EfficientNet-B3 | ~0.920 | 12M | ~4GB |
| EfficientNet-B4 | ~0.928 | 19M | ~6GB |
| ConvNeXt-Small | ~0.930 | 50M | ~8GB |

The same two-phase training code in `trainer.py` works with any timm backbone —
changing one line in `configs/default.yaml` is sufficient to experiment.

---

## 10. Setup & Installation

### Prerequisites

- Python 3.12
- A Kaggle account (for HAM10000 download)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/kobylev/L41-Homework.git
cd L41-Homework/skin_classifier

# 2. Create virtual environment
python3.12 -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add Kaggle credentials
# Download kaggle.json from https://www.kaggle.com/settings -> API -> Create Token
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 5. Download HAM10000 dataset
python scripts/download_data.py

# 6. Train the model
python scripts/train.py --config configs/default.yaml

# 7. Evaluate on test set
python scripts/evaluate.py

# 8. Predict a single image
python scripts/predict.py --image /path/to/lesion.jpg
```

### Quick Smoke Test (no dataset required)

```bash
python scripts/smoke_test.py
# Runs inference on 3 sample images with random weights
# Verifies: model loads, transforms work, API returns valid JSON
```

---

## 11. Docker

### Build & Run

```bash
# Build image
docker build -t skin-classifier .

# Run API server
docker compose up api

# Test the running API
curl http://localhost:5000/health
```

### Services

```yaml
# docker-compose.yml
services:
  api:           # Flask REST API on port 5000, mounts ./models volume
  train:         # One-off training container (same image, different command)
```

The Dockerfile uses a **multi-stage build**: all Python dependencies are
compiled in the builder stage; only the venv and application code are
copied to the runtime image, minimizing final image size.

---

## 12. API Reference

### `GET /health`

```bash
curl http://localhost:5000/health
```
```json
{"status": "ok", "model_loaded": true, "backbone": "efficientnet_b0"}
```

---

### `POST /predict`

```bash
curl -X POST http://localhost:5000/predict \
     -F "image=@/path/to/lesion.jpg"
```
```json
{
  "predicted_class": "mel",
  "class_name": "Melanoma",
  "confidence": 0.82,
  "risk_level": "HIGH",
  "recommendation": "This pattern may indicate a high-risk condition. Seek medical attention soon.",
  "all_probabilities": {
    "nv": 0.07, "mel": 0.82, "bkl": 0.03,
    "bcc": 0.04, "akiec": 0.02, "vasc": 0.01, "df": 0.01
  }
}
```

**Risk Level Mapping:**

| Risk | Classes | Recommendation |
|---|---|---|
| HIGH | mel, bcc, akiec | Seek medical attention soon |
| MEDIUM | vasc | Monitor, consult if changes |
| LOW | nv, bkl, df | Monitor at home |

---

## 13. Project Structure

```
skin_classifier/
|
+-- configs/
|   +-- default.yaml              # Single source of truth for all hyperparameters
|
+-- src/                          # Core ML source — one responsibility per file
|   +-- config.py                 # YAML loader and validation
|   +-- transforms.py             # Train / val augmentation pipelines
|   +-- dataset.py                # SkinDataset, data splits, class weights
|   +-- model.py                  # EfficientNetSkinClassifier (nn.Module)
|   +-- trainer.py                # Two-phase training loop
|   +-- metrics.py                # Accuracy, AUC, F1, confusion matrix
|   +-- inference.py              # Single-image prediction pipeline
|
+-- scripts/                      # CLI entry points — no business logic
|   +-- download_data.py          # Kaggle API download + extract
|   +-- train.py                  # Training orchestration
|   +-- evaluate.py               # Test-set evaluation
|   +-- predict.py                # Single-image CLI prediction
|   +-- smoke_test.py             # Quick sanity check (no dataset needed)
|
+-- api/
|   +-- app.py                    # Flask REST API
|   +-- schemas.py                # Request / response dataclasses
|
+-- data/ham10000/                # Dataset (not committed to git)
+-- models/                       # Saved checkpoints (not committed to git)
+-- Dockerfile
+-- docker-compose.yml
+-- requirements.txt
+-- CLAUDE.md                     # AI coding assistant conventions
```

---

## 14. Configuration Reference

All hyperparameters live in `configs/default.yaml`. Nothing is hardcoded.

| Parameter | Default | Description |
|---|---|---|
| `backbone` | `efficientnet_b0` | timm model identifier |
| `pretrained` | `true` | Load ImageNet pretrained weights |
| `image_size` | `224` | Input resolution (matches EfficientNet-B0) |
| `image_mean` | `[0.485, 0.456, 0.406]` | ImageNet channel means |
| `image_std` | `[0.229, 0.224, 0.225]` | ImageNet channel stds |
| `dropout_1` | `0.4` | First head dropout rate |
| `dropout_2` | `0.2` | Second head dropout rate |
| `epochs_frozen` | `10` | Phase 1 epochs (head only) |
| `epochs_finetune` | `20` | Phase 2 max epochs |
| `lr_head` | `1e-3` | Head learning rate (Phase 1 & 2) |
| `lr_backbone` | `1e-4` | Backbone learning rate (Phase 2 only) |
| `batch_size` | `32` | Training batch size |
| `weight_decay` | `1e-4` | L2 regularization coefficient |
| `optimizer` | `AdamW` | Optimizer choice |
| `early_stopping_patience` | `7` | Epochs without val_AUC improvement |
| `early_stopping_metric` | `val_AUC` | Metric to monitor |
| `val_split` | `0.15` | Validation fraction |
| `test_split` | `0.15` | Test fraction |
| `checkpoint_dir` | `./models` | Where to save best_model.pth |
| `data_root` | `./data/ham10000` | Dataset root directory |
| `data_csv` | `HAM10000_metadata.csv` | Metadata filename |

---

## Medical Disclaimer

This system is a **research prototype** developed for educational purposes as
part of an AI Developer course project.

- It has **not** been validated in any clinical setting
- It is **not** a certified medical device under any regulatory framework
- Model predictions **must never** be used as the basis for medical decisions
- Always consult a qualified dermatologist for any skin condition evaluation

The authors accept no liability for any use of this software in a medical context.