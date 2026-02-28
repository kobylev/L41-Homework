import os
import argparse
import pandas as pd
import torch
import sys

# Ensure src module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.dataset import prepare_dataloaders, REVERSE_CLASS_MAPPING
from src.transforms import get_train_transforms, get_val_transforms
from src.model import EfficientNetSkinClassifier
from src.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = os.path.join(cfg.data_root, cfg.data_csv)
    df = pd.read_csv(csv_path)
    
    train_transforms = get_train_transforms(cfg)
    val_transforms = get_val_transforms(cfg)
    
    # We load the data splitting just to get the test_loader
    _, _, test_loader, class_weights = prepare_dataloaders(
        cfg, df, train_transforms, val_transforms
    )
    
    model = EfficientNetSkinClassifier(
        backbone_name=cfg.backbone,
        pretrained=False,
        dropout_1=cfg.dropout_1,
        dropout_2=cfg.dropout_2
    )
    
    model_path = os.path.join(cfg.checkpoint_dir, 'best_model.pth')
    
    # Check if a trained model actually exists
    if not os.path.exists(model_path):
        print(f"Error: No trained model found at {model_path}. Please train the model first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    # Use Trainer to run evaluation (we pass test_loader as the validation loader)
    trainer = Trainer(model, train_loader=None, val_loader=test_loader, cfg=cfg, class_weights=class_weights, device=device)
    
    print(f"\nEvaluating on Test Set ({len(test_loader.dataset)} unseen images)...")
    _, metrics, _, _ = trainer._val_epoch()
    
    print("\n--- Final Test Set Results ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Macro AUC: {metrics['auc']:.4f}")
    print(f"Macro F1:  {metrics['macro_f1']:.4f}")
    print("\nPer-Class F1 Scores (Unseen Data):")
    for i in range(7):
        cls_name = REVERSE_CLASS_MAPPING[i]
        f1 = metrics['per_class_f1'][i]
        print(f"  {cls_name}: {f1:.4f}")

if __name__ == "__main__":
    main()
