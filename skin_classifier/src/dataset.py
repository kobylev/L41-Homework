import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
from sklearn.model_selection import StratifiedShuffleSplit

CLASS_MAPPING = {
    'nv': 0,
    'mel': 1,
    'bkl': 2,
    'bcc': 3,
    'akiec': 4,
    'vasc': 5,
    'df': 6
}

REVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

CLASS_NAMES = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratosis',
    'vasc': 'Vascular lesion',
    'df': 'Dermatofibroma'
}

RISK_MAPPING = {
    'mel': 'HIGH',
    'bcc': 'HIGH',
    'akiec': 'HIGH',
    'vasc': 'MEDIUM',
    'nv': 'LOW', 
    'bkl': 'LOW',
    'df': 'LOW'
}

class SkinDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: str, transforms=None):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.transforms = transforms
        
        self.image_paths = []
        for img_id in self.df['image_id']:
            paths = [
                os.path.join(self.data_root, f"{img_id}.jpg"),
                os.path.join(self.data_root, "HAM10000_images_part_1", f"{img_id}.jpg"),
                os.path.join(self.data_root, "HAM10000_images_part_2", f"{img_id}.jpg")
            ]
            found = False
            for p in paths:
                if os.path.exists(p):
                    self.image_paths.append(p)
                    found = True
                    break
            if not found:
                # Default to root if not found or not downloaded yet
                self.image_paths.append(os.path.join(self.data_root, f"{img_id}.jpg"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        label_str = self.df.loc[idx, 'dx']
        label = CLASS_MAPPING[label_str]
        
        if self.transforms:
            image = self.transforms(image)
        
        return image, torch.tensor(label, dtype=torch.long)

def get_class_weights(df: pd.DataFrame) -> torch.Tensor:
    class_counts = df['dx'].value_counts()
    weights = []
    # Assumes classes are indexed 0 to 6 in order of CLASS_MAPPING
    # Wait, CLASS_MAPPING dictionaries preserve insertion order in Python 3.7+
    for cls_name in CLASS_MAPPING.keys():
        count = class_counts.get(cls_name, 0)
        weight = 1.0 / count if count > 0 else 0.0
        weights.append(weight)
    
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() # Normalize
    return weights

def prepare_dataloaders(cfg, df: pd.DataFrame, train_transforms, val_transforms):
    # Split by lesion_id to prevent data leakage from duplicate images
    lesion_df = df.drop_duplicates(subset=['lesion_id']).reset_index(drop=True)
    
    # StratifiedShuffleSplit expects at least 2 samples per class
    # There are rare classes, but typically > 2 lesions per class in HAM10000.
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=cfg.test_split, random_state=42)
    train_val_idx, test_idx = next(sss_test.split(lesion_df, lesion_df['dx']))
    
    lesion_train_val = lesion_df.iloc[train_val_idx].reset_index(drop=True)
    lesion_test = lesion_df.iloc[test_idx].reset_index(drop=True)
    
    val_ratio = cfg.val_split / (1.0 - cfg.test_split) 
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    train_idx, val_idx = next(sss_val.split(lesion_train_val, lesion_train_val['dx']))
    
    lesion_train = lesion_train_val.iloc[train_idx].reset_index(drop=True)
    lesion_val = lesion_train_val.iloc[val_idx].reset_index(drop=True)
    
    # Get all images for corresponding lesions
    train_df = df[df['lesion_id'].isin(lesion_train['lesion_id'])].reset_index(drop=True)
    val_df = df[df['lesion_id'].isin(lesion_val['lesion_id'])].reset_index(drop=True)
    test_df = df[df['lesion_id'].isin(lesion_test['lesion_id'])].reset_index(drop=True)
    
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(test_df)}")
    
    train_dataset = SkinDataset(train_df, cfg.data_root, transforms=train_transforms)
    val_dataset = SkinDataset(val_df, cfg.data_root, transforms=val_transforms)
    test_dataset = SkinDataset(test_df, cfg.data_root, transforms=val_transforms)
    
    # Calculate sampling weights (inverse frequency) for minority class oversampling
    class_weights = get_class_weights(train_df)
    sample_weights = [class_weights[CLASS_MAPPING[label]] for label in train_df['dx']]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        sampler=sampler, 
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_weights
