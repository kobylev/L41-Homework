import torch
import torch.nn as nn
import timm

class EfficientNetSkinClassifier(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0", pretrained=True, dropout_1=0.4, dropout_2=0.2, num_classes=7):
        super().__init__()
        # Load pre-trained EfficientNet backbone
        # Transfer Learning Rationale:
        # ImageNet features transfer well to dermatology:
        # - Early layers: detect texture gradients -> useful for lesion borders and edges
        # - Middle layers: detect color patterns -> useful for pigmentation analysis
        # - Late layers: detect complex shapes -> useful for overall lesion structure
        
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0) # num_classes=0 removes last fc layer
        num_features = self.backbone.num_features # 1280 for b0
        
        # Classification head architecture:
        # EfficientNet-B0 backbone (1280-dim features)
        # → Linear(1280, 512) + ReLU + Dropout(dropout_1)
        # → Linear(512, 128) + ReLU + Dropout(dropout_2)
        # → Linear(128, 7)
        self.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x) # [B, 1280]
        out = self.head(features)
        return out
