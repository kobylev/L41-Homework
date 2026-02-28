import torchvision.transforms as T
from .config import Config

def get_train_transforms(cfg: Config):
    return T.Compose([
        T.Resize((cfg.image_size, cfg.image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=20),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ToTensor(),
        T.Normalize(mean=cfg.image_mean, std=cfg.image_std)
    ])

def get_val_transforms(cfg: Config):
    return T.Compose([
        T.Resize((cfg.image_size, cfg.image_size)),
        T.ToTensor(),
        T.Normalize(mean=cfg.image_mean, std=cfg.image_std)
    ])
