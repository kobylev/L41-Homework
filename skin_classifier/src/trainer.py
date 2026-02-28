import torch
import torch.nn as nn
from tqdm import tqdm
import os
import copy
from .metrics import compute_metrics, plot_confusion_matrix
from .config import Config

class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg: Config, class_weights: torch.Tensor, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        
        # Two-phase strategy tackles Severe Class Imbalance (with sampler)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        
        self.best_auc = 0.0
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.epochs_no_improve = 0
        
    def _train_epoch(self, optimizer):
        self.model.train()
        running_loss = 0.0
        y_true, y_pred, y_prob = [], [], []
        
        pbar = tqdm(self.train_loader, desc="Training")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.detach().cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(self.train_loader.dataset)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        return epoch_loss, metrics

    def _val_epoch(self):
        self.model.eval()
        running_loss = 0.0
        y_true, y_pred, y_prob = [], [], []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())
                
        epoch_loss = running_loss / len(self.val_loader.dataset)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        return epoch_loss, metrics, y_true, y_pred

    def fit(self):
        # Phase 1: Head only (backbone fully frozen)
        # Rationale: Backbone already encodes texture/color/edge features from ImageNet 
        # that directly apply to skin lesion analysis. We freeze it to avoid destroying 
        # these useful representations while training the newly initialized classification head.
        print("\n--- PHASE 1: Training Head Only ---")
        for param in self.model.backbone.parameters():
            param.requires_grad = False
            
        optimizer_head = torch.optim.AdamW(
            self.model.head.parameters(), 
            lr=self.cfg.lr_head, 
            weight_decay=self.cfg.weight_decay
        )
        
        for epoch in range(self.cfg.epochs_frozen):
            print(f"\nEpoch {epoch+1}/{self.cfg.epochs_frozen} (Phase 1)")
            train_loss, train_metrics = self._train_epoch(optimizer_head)
            val_loss, val_metrics, _, _ = self._val_epoch()
            
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.4f} | AUC: {train_metrics['auc']:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f} | AUC: {val_metrics['auc']:.4f}")
            
            self._check_early_stopping(val_metrics['auc'])

        # Phase 2: Fine-tuning (unfreeze top layers)
        # Rationale: Allow top layers to specialize for specific dermatological features
        # (e.g. distinguishing nevus vs melanoma) by unlocking high-level representation blocks.
        # We use a 10x lower learning rate to protect pretrained weights.
        print("\n--- PHASE 2: Fine-tuning Top Layers ---")
        for param in self.model.backbone.parameters():
            param.requires_grad = False
            
        if hasattr(self.model.backbone, 'blocks'):
            for block in self.model.backbone.blocks[-3:]:
                for param in block.parameters():
                    param.requires_grad = True
        
        if hasattr(self.model.backbone, 'conv_head'):
            for param in self.model.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.model.backbone, 'bn2'):
            for param in self.model.backbone.bn2.parameters():
                param.requires_grad = True
                
        optimizer_ft = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.cfg.lr_backbone, 
            weight_decay=self.cfg.weight_decay
        )
        
        self.epochs_no_improve = 0 # reset patience for phase 2
        for epoch in range(self.cfg.epochs_finetune):
            print(f"\nEpoch {epoch+1}/{self.cfg.epochs_finetune} (Phase 2)")
            train_loss, train_metrics = self._train_epoch(optimizer_ft)
            val_loss, val_metrics, _, _ = self._val_epoch()
            
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.4f} | AUC: {train_metrics['auc']:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f} | AUC: {val_metrics['auc']:.4f}")
            
            if self._check_early_stopping(val_metrics['auc']):
                print(f"Early stopping triggered in Phase 2 at epoch {epoch+1}.")
                break
                
        print("\nTraining complete. Loading best model weights.")
        self.model.load_state_dict(self.best_model_wts)
        
        _, final_val_metrics, y_true, y_pred = self._val_epoch()
        
        cm_path = os.path.join(self.cfg.checkpoint_dir, "confusion_matrix.png")
        plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
        print(f"Saved confusion matrix: {cm_path}")
        
        return final_val_metrics

    def _check_early_stopping(self, current_auc):
        if current_auc > self.best_auc:
            self.best_auc = current_auc
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
            self.epochs_no_improve = 0
            
            os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
            best_model_path = os.path.join(self.cfg.checkpoint_dir, 'best_model.pth')
            torch.save(self.model.state_dict(), best_model_path)
            print(f"--> Saved best model with val_AUC: {current_auc:.4f}")
            return False
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.cfg.early_stopping_patience:
                return True
            return False
