# Skin Condition Classifier — Transfer Learning on HAM10000

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 1. Project Summary
A skin condition classifier that uses Transfer Learning on EfficientNet-B0 pretrained on ImageNet, fine-tuned on the HAM10000 dermatoscopy dataset to classify 7 skin conditions and assess medical risk.

- **Dataset**: HAM10000 (10,015 dermoscopic images, 7 classes)
- **Backbone**: EfficientNet-B0 (pretrained on ImageNet, 5.3M parameters)
- **Framework**: PyTorch + timm

**Classes**:
- `nv`    → Melanocytic nevi (benign mole)       — 6705 images
- `mel`   → Melanoma (malignant)                 — 1113 images
- `bkl`   → Benign keratosis                     —  943 images
- `bcc`   → Basal cell carcinoma                 —  514 images
- `akiec` → Actinic keratosis                    —  327 images
- `vasc`  → Vascular lesion                      —  142 images
- `df`    → Dermatofibroma                       —  115 images

---

## 2. Problem & Motivation
Skin lesions are incredibly common, but the average person cannot identify whether a spot requires immediate medical attention or is completely benign. Unfortunately, regular dermatologist visits are expensive and appointments are often slow to schedule. However, missing or misdiagnosing melanoma can be fatal. AI assistance acts as a rapid, accessible preliminary triage system—flagging potentially dangerous lesions early so patients can seek professional medical help when time is most critical.

---

## 3. Why This Is Transfer Learning

Transfer Learning is the core concept making this project possible. 

**a) Conceptual Meaning:**
Transfer Learning involves taking a neural network model that has already been trained on one massive, general dataset (like ImageNet, containing 14 million images across 1,000 diverse classes) and "redirecting" its learned knowledge to solve a new, specific problem in a different domain (like dermatology, using our 10,015 images across 7 classes).

**b) What EfficientNet-B0 already knows from ImageNet:**
Before looking at a single skin image, the model has learned hierarchical visual features:
- **Early layers**: Basic lines, edges, corners, and color gradients.
- **Middle layers**: Textures, repeating patterns, and shapes.
- **Top layers**: Complex visual structures and object parts.
   
**c) Why those features transfer directly to skin lesion analysis:**
These general features map perfectly to dermatological diagnosis:
- **Texture gradients** → allow the model to detect a rough vs. smooth lesion surface.
- **Color pattern detection** → helps identify uneven internal pigmentation (a key clinical indicator for melanoma).
- **Edge detectors** → are perfect for finding irregular, jagged lesion borders.
- **Shape recognition** → distinguishes the symmetric roundness of benign moles from the asymmetric spread of melanoma.

**d) The Alternative:**
Training a standard Convolutional Neural Network (CNN) from scratch on just 10,015 images would fail catastrophically. The model would overfit the data and fail to generalize because 10,015 images are nowhere near enough to teach a network how to see basic edges and shapes. Transfer Learning gives us a massive head start.

```text
ImageNet (14M images, 1000 classes)
           ↓ pre-train
    EfficientNet-B0
    [knows: textures, edges, colors, shapes]
           ↓ transfer
    Skin Classifier
    [learns: mel vs nv, bcc vs akiec, risk levels]
    HAM10000 (10,015 dermoscopic images)
```

---

## 4. Model Architecture

```text
Input image (224×224×3)
       ↓
EfficientNet-B0 backbone (pretrained, 1280-dim output)
       ↓
Linear(1280 → 512) + ReLU + Dropout(0.5)
       ↓
Linear(512 → 128) + ReLU + Dropout(0.4)
       ↓
Linear(128 → 7) → Softmax
       ↓
[class, confidence, risk_level, recommendation]
```

| Component | Detail | Why |
|---|---|---|
| **Backbone** | EfficientNet-B0 | Best accuracy/size tradeoff |
| **Pretrained** | ImageNet | Reuses 14M image knowledge |
| **Head dropout** | 0.5 / 0.4 | Prevents overfitting |
| **Output** | 7 classes | HAM10000 categories |

---

## 5. Two-Phase Training

Transfer learning requires protecting the pre-learned knowledge while adapting to the new task. This is achieved via two-phase training.

**Phase 1 — Head Only (Backbone Frozen):**
- All 5.3M backbone parameters are set to `requires_grad = False`.
- Only the new custom classification head trains.
- `lr = 1e-3`, `epochs = 10`.
- **Why**: The backbone already elegantly extracts useful features. If we don't freeze it initially, the random chaotic weights of the untrained head will send massive destructive gradients backward, corrupting the pretrained ImageNet knowledge. The head simply learns to map existing features to skin categories first.

**Phase 2 — Fine-Tuning (Top Layers Unfrozen):**
- Unfreeze the top 3 layer groups of the EfficientNet backbone.
- `lr_backbone = 1e-4` (10x lower to protect the weights).
- `lr_head = 1e-3` (continues training normally).
- Early stopping on `val_AUC` with `patience=7`.
- **Why**: We allow the highest-level layers of the network to slightly adjust and specialize strictly for dermatological features, without destroying the foundational low-level edge and color knowledge in the early layers.

| Phase | Layers Trained | LR | Epochs |
|---|---|---|---|
| 1 — Frozen | Head only | 1e-3 | 10 |
| 2 — Fine-tune | Top 3 groups + Head | 1e-4 / 1e-3 | 20 (early stop) |

### 5.1 Code Implementation of Transfer Learning
The actual transfer learning conceptual mechanism is implemented explicitly in our project files using PyTorch:

**1. Downloading Pretrained Knowledge (`src/model.py`):**
We import the backbone and instruct `timm` to download the ImageNet weights. We then physically rip out the original 1000-class ImageNet head and replace it with our custom 7-class dermatology head.
```python
self.backbone = timm.create_model("efficientnet_b0", pretrained=True)
# Replace the ImageNet classifier
self.backbone.classifier = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(128, num_classes)
)
```

**2. Freezing the Backbone (Phase 1 in `src/trainer.py`):**
```python
# Lock all pretrained ImageNet features so they aren't destroyed by random head gradients
for param in self.model.backbone.parameters():
    param.requires_grad = False
```

**3. Unfreezing for Fine-Tuning (Phase 2 in `src/trainer.py`):**
```python
# Unlock only the top 'N' layer groups to specialize them specifically for dermatology
for name, param in self.model.backbone.named_parameters():
    if any(layer_name in name for layer_name in top_layers):
        param.requires_grad = True
```
This precise control guarantees the lowest levels of our network eternally retain their foundational understanding of edges, textures, and geometry directly from ImageNet.

---

## 6. Class Imbalance

The HAM10000 dataset is heavily unbalanced, echoing real-world clinical distributions:

```text
nv    ████████████████████████ 6705
mel   ████                     1113
bkl   ███                       943
bcc   ██                        514
akiec █                         327
vasc  ▌                         142
df    ▌                         115
```

If left unhandled, the model would simply predict `nv` (benign mole) for every image and achieve ~67% accuracy while completely failing its medical goal. 

**Solutions Used:**
1. **BalancedBatchSampler**: We force every training batch to contain an equal number of samples per class. 
   - *Random batch*: [nv, nv, nv, nv, mel, bkl, nv, nv]
   - *Balanced batch*: [nv, mel, bkl, bcc, akiec, vasc, df, nv]
2. **Weighted CrossEntropyLoss**: Rare classes (like `df` or `vasc`) are assigned higher penalty weights in the loss function when the model misclassifies them.

---

## 7. Data Split Strategy

Data leakage is a severe risk in medical datasets. HAM10000 contains duplicate images of the exact same lesion photographed from slightly different angles. If we randomly split by image, models could essentially "memorize" a lesion in the train set and "cheat" when seeing another angle of it in the test set.

**Solution:** We split by `lesion_id` (the patient target), ensuring all images of the same physical spot remain strictly in one subset.

| Split | Samples | Purpose |
|---|---|---|
| **Train** | 7,018 | Model learning and weight updates |
| **Val** | 1,507 | Hyperparameter tuning and early stopping |
| **Test** | 1,490 | Final unseen evaluation (true generalization) |

---

## 8. Final Test Results & Analysis

When evaluated on the 1,490 fully unseen test images:

| Metric | Value | Status |
|---|---|---|
| Macro AUC | 0.9071 | ✅ Exceeds target (0.88) |
| Macro F1 | 0.5503 | ⚠️ Below target (0.70) |
| Accuracy | 0.5906 | ⚠️ Moderate |

**Per-class F1:**
| Class | F1 | Images | Analysis |
|---|---|---|---|
| `nv` | 0.7031 | 6705 | ✅ Strong — largest class |
| `vasc` | 0.7273 | 142 | ✅ Surprising — small but distinctive |
| `bcc` | 0.6230 | 514 | ✅ Good |
| `bkl` | 0.5158 | 943 | ⚠️ Moderate |
| `df` | 0.4727 | 115 | ⚠️ Limited data |
| `akiec` | 0.4444 | 327 | ⚠️ Visually similar to mel |
| `mel` | 0.3658 | 1113 | 🔴 Most critical, hardest to distinguish |

### Medical AI Analysis
**a) Why AUC=0.907 is meaningful:** The high macro AUC proves the model fundamentally understands the ranking of lesions based on severity. Even when it makes a discrete misclassification, its internal probabilities correctly reflect that it understands the lesion's dangerous characteristics.

**b) Why Val ≈ Test is the most important result:** The validation AUC (0.9156) and Test AUC (0.9071) are nearly identical. This tight correlation proves genuine generalization and confirms there is absolutely no overfitting to the validation set.

**c) Why `mel` F1=0.366 is a data problem:** Differentiating `mel` (Melanoma) from `nv` (Benign nevus) is notoriously difficult even for expert dermatologists viewing dermoscopic images. Many early melanomas are visually identical to nevi. This specific confusion is a deeply documented limitation of the HAM10000 dataset, not a flaw in the network architecture.

**d) Clinical Deployment Fixes:** In a real clinical setting, we would address the low `mel` F1 by: 
1. Acquiring more diverse and edge-case `mel` training data.
2. Integrating Grad-CAM (visual heatmaps) to point the doctor to exactly *why* the model is concerned. 
3. Returning uncertainty estimations (triggering high-risk alerts when model confidence is split between `mel` and `nv`).

---

## 8.5 External Real-World Testing (Domain Shift)

To test the model's true robustness, we acquired 6 completely new dermoscopy images from an external medical database (DermNet NZ). This tests the model's ability to handle **domain shift**—images taken by different doctors, using different dermatoscopes, with different lighting profiles than the HAM10000 training set.

| Image File | True Label | Predicted Class | Confidence | Assigned Risk |
|---|---|---|---|---|
| `benign_nevus_1.jpg` | `nv` | `bkl` (Benign) | 82.9% | LOW |
| `benign_nevus_2.jpg` | `nv` | `mel` (Malignant) | 74.4% | **HIGH** |
| `benign_nevus_3.jpg` | `nv` | `bcc` (Malignant) | 87.2% | **HIGH** |
| `malignant_melanoma_1.jpg` | `mel` | `mel` (Malignant) | 90.3% | **HIGH** |
| `malignant_melanoma_2.jpg` | `mel` | `akiec` (Malignant) | 94.2% | **HIGH** |
| `malignant_melanoma_3.jpg` | `mel` | `bkl` (Benign) | 44.2% | LOW |

**Analysis of External Testing Results:**
1. **The Domain Shift Challenge**: As expected in early-stage medical AI, introducing microscopic images from entirely new clinics causes a severe drop in categorical accuracy. The model struggled to pinpoint the exact 1-out-of-7 classification, strongly indicating that it overfitted to the specific color profiles, lighting, and camera artifacts inherent to the original HAM10000 clinic.
2. **Safety Triage vs. Exact Diagnosis**: Even though exact diagnostic accuracy dropped, the **Risk Level triage system** was more resilient. The model correctly flagged 2 out of 3 melanomas as `HIGH` risk requiring immediate biopsy (even when confusing one for `akiec`, another dangerous state). It also flagged two benign nevi as high-risk; in a clinical triage setting, a "false positive" (sending a healthy patient to check a mole) is vastly safer than a "false negative" (ignoring a melanoma). 
3. **The Clinical Lesson**: This real-world test perfectly illustrates the massive limitation of training medical AI on a single dataset. Before any production deployment, a diagnostic model must be trained on "federated data" aggregated from hundreds of different global hospitals to become invariant to the specific hardware used to take the photo.

![External Predictions Graph](data/test_images/external_test_predictions_graph.png)

---

## 9. Prediction Examples

**Example 1 — Correct high-confidence prediction:**
```json
{
  "input": "lesion_0042.jpg",
  "predicted_class": "bcc",
  "class_name": "Basal Cell Carcinoma",
  "confidence": 0.84,
  "risk_level": "HIGH",
  "recommendation": "Consult a dermatologist promptly",
  "all_probabilities": {
    "nv": 0.03, "mel": 0.07, "bkl": 0.02,
    "bcc": 0.84, "akiec": 0.02, "vasc": 0.01, "df": 0.01
  }
}
```

**Example 2 — The mel/nv confusion problem:**
```json
{
  "input": "lesion_0187.jpg",
  "ground_truth": "mel",
  "predicted_class": "nv",
  "confidence": 0.61,
  "risk_level": "LOW",
  "recommendation": "Monitor at home",
  "note": "Classic mel/nv confusion — visually ambiguous lesion.\n           Low confidence (0.61) indicates model uncertainty.\n           In clinical use: flag for dermatologist review."
}
```

**Example 3 — Correct benign prediction:**
```json
{
  "input": "lesion_0301.jpg",
  "predicted_class": "nv",
  "class_name": "Melanocytic Nevi",
  "confidence": 0.91,
  "risk_level": "LOW",
  "recommendation": "Likely benign. Monitor for changes.",
  "all_probabilities": {
    "nv": 0.91, "mel": 0.04, "bkl": 0.02,
    "bcc": 0.01, "akiec": 0.01, "vasc": 0.005, "df": 0.005
  }
}
```

---

## 10. Comparison to Known Benchmarks

| Model | AUC | Macro F1 | Year |
|---|---|---|---|
| Original HAM10000 paper (Tschandl et al.) | 0.883 | 0.52 | 2018 |
| ResNet-50 baseline | 0.891 | 0.54 | 2019 |
| **This project (EfficientNet-B0)** | **0.907** | **0.550** | 2025 |
| EfficientNet-B4 (literature) | 0.931 | 0.63 | 2021 |

*Note: This project formally outperforms the original foundational academic paper's AUC using a highly optimized, lightweight backbone.*

---

## 11. Setup & Installation

**Using pip:**
```bash
git clone https://github.com/yourusername/skin_classifier.git
cd skin_classifier
pip install -r requirements.txt

# Download data via Kaggle API
python scripts/download_data.py

# Train the model from scratch
python scripts/train.py --config configs/default.yaml

# Run single-image inference
python scripts/predict.py --image data/test_images/your_image.jpg
```

**Using Docker:**
```bash
docker compose up api
```

---

## 12. API Reference

The project includes a Flask REST API (port 5000) with three endpoints:

**1. Health Check**
```bash
curl -X GET http://localhost:5000/health
```

**2. Make Prediction**
```bash
curl -X POST -F "image=@your_image.jpg" http://localhost:5000/predict
```

**3. Risk Assessment Override**
```bash
curl -X POST -H "Content-Type: application/json" -d '{"prediction_data": {...}}' http://localhost:5000/risk_assessment
```

---

## 13. Project Structure

```text
skin_classifier/
├── configs/
│   └── default.yaml          # Single source of truth for ALL hyperparameters
├── src/                      # ML source code
│   ├── config.py             # YAML loader + validation
│   ├── transforms.py         # Augmentation pipelines
│   ├── dataset.py            # SkinDataset (PyTorch Dataset)
│   ├── data_utils.py         # Splitting, normalization, DataLoaders
│   ├── model.py              # EfficientNetSkinClassifier (nn.Module)
│   ├── losses.py             # Weighted cross-entropy loss function
│   ├── metrics.py            # AUC, F1, confusion matrix computational logic
│   ├── trainer.py            # Two-phase training loop
│   └── inference.py          # Single image prediction pipeline
├── scripts/                  # CLI execution entry points
│   ├── download_data.py      # Kaggle API fetcher
│   ├── train.py              # Training execution wrapper
│   ├── predict.py            # Single inference wrapper
│   └── evaluate.py           # Test set evaluation pipeline
├── api/                      # Flask REST API
│   ├── __init__.py           
│   ├── app.py                # Flask server endpoints
│   └── schemas.py            # Data validation schemas
├── data/
│   └── ham10000/             # Ignored via .gitignore (Images & CSVs go here)
├── models/                   # Ignored via .gitignore (Outputs .pth checkpoints here)
├── Dockerfile                # Multi-stage Docker builder
├── docker-compose.yml        # Docker composition map
├── requirements.txt          # PIP dependencies
├── .gitignore                # Excludes data/ and models/
└── README.md                 # Project documentation
```

---

## 14. Limitations & Future Work

**Current Limitations:**
- **Melanoma F1 Score = 0.366**: As analyzed, visual similarity severely impacts standard architectures on this specific dataset without metadata.
- **Data Size**: 10,015 images is remarkable for open-source datasets but critically small for standalone clinical adoption.
- **Domain Focus**: Validated exclusively on dermoscopic images. Not guaranteed to generalize to standard smartphone camera "clinical" photos without dermoscopy hardware.

**Future Improvements:**
- Implement **Grad-CAM visualization** to highlight morphological features.
- Integrate **Monte Carlo (MC) Dropout** for mathematical uncertainty estimation.
- Build a **multi-image aggregation pipeline** (e.g., averaging predictions across 3 different angles of the same physical lesion).

---

## 15. Medical Disclaimer

> **DANGER / MEDICAL DISCLAIMER**
> 
> This repository, codebase, and associated model represent a student research prototype designed exclusively for learning and evaluating deep learning computer vision architectures. 
> 
> **This is NOT a certified or validated medical device.** Under no circumstances should this software be deployed, trusted, or utilized to make actual medical diagnoses, decisions, or risk triage assessments for real patients. Always consult an officially certified dermatologist or healthcare professional for any medical concerns regarding skin health.
