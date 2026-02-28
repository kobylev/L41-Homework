import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from .dataset import REVERSE_CLASS_MAPPING

def compute_metrics(y_true, y_pred, y_prob):
    # Overall accuracy
    acc = accuracy_score(y_true, y_pred)
    
    # Macro F1
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Weighted AUC (one-vs-rest, multi-class)
    try:
        # y_prob should be [n_samples, n_classes]
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted', labels=list(range(7)))
    except ValueError:
        auc = 0.0 # If a class is completely missing in y_true, this handles the exception
        
    # Per-class metrics
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(7)), zero_division=0)
    
    metrics = {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'auc': auc,
        'per_class_f1': f1_per_class,
        'per_class_precision': precision,
        'per_class_recall': recall
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred, save_path="models/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(7)))
    plt.figure(figsize=(10, 8))
    
    # Class names for labels
    labels = [REVERSE_CLASS_MAPPING[i] for i in range(7)]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
