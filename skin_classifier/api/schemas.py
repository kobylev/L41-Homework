from dataclasses import dataclass
from typing import Dict

@dataclass
class PredictionResponse:
    predicted_class: str
    class_name: str
    confidence: float
    risk_level: str
    recommendation: str
    all_probabilities: Dict[str, float]

@dataclass
class HealthResponse:
    status: str
    model_loaded: bool
    backbone: str
