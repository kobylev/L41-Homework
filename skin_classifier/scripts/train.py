import os
import argparse
import pandas as pd
import torch
import sys

# Ensure src module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.dataset import prepare_dataloaders
from src.transforms import get_train_transforms, get_val_transforms
from src.model import EfficientNetSkinClassifier
from src.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load metadata
    csv_path = os.path.join(cfg.data_root, cfg.data_csv)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata CSV not found at {csv_path}. Please run download_data.py first.")
        
    df = pd.read_csv(csv_path)
    
    # Transforms
    train_transforms = get_train_transforms(cfg)
    val_transforms = get_val_transforms(cfg)
    
    # Dataloaders
    train_loader, val_loader, test_loader, class_weights = prepare_dataloaders(
        cfg, df, train_transforms, val_transforms
    )
    
    # Model
    model = EfficientNetSkinClassifier(
        backbone_name=cfg.backbone,
        pretrained=cfg.pretrained,
        dropout_1=cfg.dropout_1,
        dropout_2=cfg.dropout_2
    )
    
    # Trainer
    trainer = Trainer(model, train_loader, val_loader, cfg, class_weights, device)
    
    # Train
    print("Starting training pipeline...")
    final_metrics = trainer.fit()
    
    print("\n--- Training Pipeline Finished ---")
    print(f"Final Validation AUC: {final_metrics['auc']:.4f}")
    print(f"Final Validation Macro F1: {final_metrics['macro_f1']:.4f}")
    
    from src.dataset import REVERSE_CLASS_MAPPING
    for i in range(7):
        cls_name = REVERSE_CLASS_MAPPING[i]
        f1 = final_metrics['per_class_f1'][i]
        print(f"Class {cls_name} F1: {f1:.4f}")

if __name__ == "__main__":
    main()
