import yaml
from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    # Model Architecture
    backbone: str
    pretrained: bool
    image_size: int
    image_mean: List[float]
    image_std: List[float]
    dropout_1: float
    dropout_2: float

    # Training Hyperparameters
    epochs_frozen: int
    epochs_finetune: int
    lr_head: float
    lr_backbone: float
    batch_size: int
    weight_decay: float
    optimizer: str
    early_stopping_patience: int
    early_stopping_metric: str

    # Paths and Data Settings
    checkpoint_dir: str
    save_best_only: bool
    data_root: str
    data_csv: str

    # Dataset Splitting
    val_split: float
    test_split: float
    num_workers: int

    @classmethod
    def load(cls, config_path: str = "configs/default.yaml") -> 'Config':
        """Load configuration from a YAML file."""
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

if __name__ == "__main__":
    cfg = Config.load()
    print("Configuration loaded successfully!")
    print(cfg)
