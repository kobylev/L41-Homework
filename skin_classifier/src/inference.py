import torch
from torchvision import transforms
from PIL import Image
import os
import sys

# Ensure src module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import EfficientNetSkinClassifier
from src.config import Config
from src.dataset import CLASS_NAMES, REVERSE_CLASS_MAPPING, RISK_MAPPING

class SkinConditionPredictor:
    def __init__(self, config_path="configs/default.yaml", model_path="models/best_model.pth"):
        self.cfg = Config.load(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = EfficientNetSkinClassifier(
            backbone_name=self.cfg.backbone,
            pretrained=False,
            dropout_1=self.cfg.dropout_1,
            dropout_2=self.cfg.dropout_2
        )
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(f"Warning: Model weights not found at {model_path}. Prediction will be random.")
            
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((self.cfg.image_size, self.cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cfg.image_mean, std=self.cfg.image_std)
        ])
        
    def predict(self, image: Image.Image) -> dict:
        image = image.convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
            
        pred_idx = probs.argmax()
        pred_class_short = REVERSE_CLASS_MAPPING[pred_idx]
        pred_class_full = CLASS_NAMES[pred_class_short]
        confidence = float(probs[pred_idx])
        risk_level = RISK_MAPPING[pred_class_short]
        
        if risk_level == "HIGH":
            recommendation = "This pattern may indicate a high-risk condition (seek medical attention soon)."
        elif risk_level == "MEDIUM":
            recommendation = "Monitor and consult a doctor if you notice any changes in size, shape, or color."
        else:
            recommendation = "Likely benign, but continue to monitor it at home."
            
        all_probs = {REVERSE_CLASS_MAPPING[i]: float(probs[i]) for i in range(7)}
            
        result = {
            "predicted_class": pred_class_short,
            "class_name": pred_class_full,
            "confidence": confidence,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "all_probabilities": all_probs
        }
        
        return result
