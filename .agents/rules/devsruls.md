---
trigger: always_on
---

Create a CLAUDE.md file for a skin condition classifier project.
Base the rules strictly on the following project conventions, 
derived from the alienspirit7/L41_HomeWork structure.

## Project Identity
- Project name: skin_classifier
- Language: Python 3.12
- Framework: PyTorch + timm
- Backbone: EfficientNet-B0 pretrained on ImageNet

---

## CLAUDE.md content to generate:

### 1. Project Structure Rules
Enforce this exact folder layout — never deviate:

skin_classifier/
├── configs/
│   └── default.yaml          # single source of truth for ALL hyperparameters
├── src/                      # all ML source code lives here
│   ├── __init__.py
│   ├── config.py             # YAML loader + validation only
│   ├── transforms.py         # augmentation pipelines only
│   ├── dataset.py            # SkinDataset (PyTorch Dataset) only
│   ├── data_utils.py         # splitting, normalization, DataLoaders only
│   ├── model.py              # EfficientNetSkinClassifier (nn.Module) only
│   ├── losses.py             # weighted cross-entropy loss only
│   ├── metrics.py            # accuracy, AUC, F1, confusion matrix only
│   ├── trainer.py            # two-phase training loop only
│   └── inference.py          # single image prediction pipeline only
├── scripts/                  # CLI entry points only — no logic here
│   ├── download_data.py
│   ├── train.py
│   └── predict.py
├── api/                      # Flask REST API only
│   ├── __init__.py
│   ├── app.py
│   └── schemas.py
├── data/
│   └── ham10000/             # never commit dataset files
├── models/                   # never commit .pth files
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── .gitignore
└── README.md

### 2. Module Responsibility Rules
Each src/ file has ONE responsibility. Never mix concerns:
- config.py      → load and validate YAML only
- transforms.py  → image transforms only, no model code
- dataset.py     → __getitem__ and __len__ only, no training logic
- data_utils.py  → splits and DataLoaders only, no model code
- model.py       → nn.Module definition only, no training loop
- losses.py      → loss functions only
- metrics.py     → metric computation only, no I/O
- trainer.py     → training loop only, calls other modules
- inference.py   → prediction pipeline only, no training code

### 3. Configuration Rules
- ALL hyperparameters must live in configs/default.yaml
- NEVER hardcode any number in src/ or scripts/
- Every script receives config via: cfg = load_config("configs/default.yaml")
- Required config keys:
  backbone, pretrained, freeze_backbone, unfreeze_top_n,
  image_size, image_mean, image_std,
  epochs_frozen, epochs_finetune,
  lr_head, lr_backbone, weight_decay, optimizer,
  batch_size, num_workers, val_split, test_split,
  loss_type, dropout_1, dropout_2,
  early_stopping_patience, early_stopping_metric,
  checkpoint_dir, save_best_only,
  data_csv, data_root

### 4. Transfer Learning Rules
Two-phase training is mandatory — never train end-to-end from epoch 1:

Phase 1 (frozen):
  - Set ALL backbone parameters: requires_grad = False
  - Train classification head only
  - Use lr_head from config
  - Run for epochs_frozen epochs

Phase 2 (fine-tune):
  - Unfreeze top unfreeze_top_n layer groups only
  - Use lr_backbone (must be 10x lower than lr_head)
  - Early stopping monitors early_stopping_metric
  - Restore best checkpoint when stopping

### 5. Data Rules
- Split by lesion_id (not by image) — prevent data leakage
- Use StratifiedShuffleSplit to preserve class distribution
- Compute class weights from TRAINING SET ONLY
- Apply ImageNet normalization:
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
- Never apply random augmentations during validation or inference

### 6. API Rules (mirrors alienspirit7 api/ structure)
Three endpoints only:

GET  /health
  → {"status": "ok", "model_loaded": true, "backbone": "efficientnet_b0"}

POST /predict
  → input:  multipart image upload
  → output: predicted_class, class_name, confidence,
            risk_level, recommendation, all_probabilities

POST /risk_assessment
  → input:  JSON with predicted probabilities + optional patient metadata
  → output: updated risk_level based on combined factors
  → mirrors alienspirit7's /effective_carbs endpoint concept:
    a second endpoint that recalculates the final recommendation
    without re-running inference

### 7. Docker Rules
Multi-stage build only (mirrors alienspirit7/Dockerfile):
  Stage 1 (builder): install all deps into /venv
  Stage 2 (runtime): copy /venv + app code only

docker-compose.yml must define exactly two services:
  - api:   Flask server, port 5000, mounts ./models volume
  - train: one-off training container (same image, different command)

### 8. Code Style Rules
- No placeholder code — every function must be fully implemented
- No TODO comments in submitted code
- Every module must have a docstring explaining its single responsibility
- model.py must include a comment block explaining WHAT ImageNet features
  transfer to skin lesion analysis (texture → lesion borders,
  color patterns → pigmentation, edge detectors → lesion shape)
- trainer.py must include inline comments explaining WHY each training
  phase exists (not just what it does)

### 9. .gitignore Rules
Always exclude:
  data/
  models/
  venv/
  __pycache__/
  *.pth
  *.pt
  *.egg-info
  .env
  .DS_Store

### 10. README Rules (mandatory sections)
README must contain ALL of these sections in order:
  1. Problem & motivation
  2. Project structure (copy the tree above)
  3. Architecture overview (text-based ASCII diagram)
  4. Data flow — end-to-end pipeline
  5. Transfer learning explanation
     (what ImageNet learned → why it transfers → two-phase rationale)
  6. Class imbalance handling
  7. Setup instructions
  8. Docker & deployment
  9. Data acquisition (Kaggle download steps)
  10. Training the model — step by step
  11. Evaluating the model (metrics table)
  12. Using the model — inference
  13. REST API reference (curl examples)
  14. Configuration reference (full table of all YAML keys)
  15. Medical disclaimer

---

Generate the complete CLAUDE.md file following ALL rules above.
The file should be written so that an AI coding assistant 
reading it will produce code that exactly matches the 
alienspirit7/L41_HomeWork architectural conventions.