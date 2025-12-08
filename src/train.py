"""
Training Script for Deepfake Detection Model

Provides complete training pipeline with:
- Training loop with validation
- Early stopping
- Model checkpointing
- Metrics logging
- Learning rate scheduling
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from module3_model import DeepfakeDetector, create_data_loaders, save_model, load_model
from gradcam import GradCAM, generate_comparison_grid
from utils import (set_seed, get_device, create_directory, save_figure,
                   AverageMeter, EarlyStopping, MetricsLogger)


def train_one_epoch(model: nn.Module,
                    train_loader: torch.utils.data.DataLoader,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int) -> dict:
    """
    Train for one epoch.
    
    Returns:
        Dictionary of training metrics
    """
    model.train()
    
    losses = AverageMeter()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.float().to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        
        # Handle single sample batch
        if outputs.dim() == 0:
            outputs = outputs.unsqueeze(0)
        
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        losses.update(loss.item(), images.size(0))
        probs = torch.sigmoid(outputs).detach()
        preds = (probs > 0.5).long()
        
        correct += (preds == labels.long()).sum().item()
        total += labels.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        
        pbar.set_postfix({'loss': losses.avg, 'acc': correct / total})
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics = {
        'train_loss': losses.avg,
        'train_acc': accuracy_score(all_labels, all_preds),
        'train_precision': precision_score(all_labels, all_preds, zero_division=0),
        'train_recall': recall_score(all_labels, all_preds, zero_division=0),
        'train_f1': f1_score(all_labels, all_preds, zero_division=0),
    }
    
    if len(np.unique(all_labels)) > 1:
        metrics['train_auc'] = roc_auc_score(all_labels, all_probs)
    else:
        metrics['train_auc'] = 0.0
    
    return metrics


def validate(model: nn.Module,
             val_loader: torch.utils.data.DataLoader,
             criterion: nn.Module,
             device: torch.device,
             epoch: int) -> dict:
    """
    Validate model.
    
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    
    losses = AverageMeter()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.float().to(device)
            
            outputs = model(images).squeeze()
            
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            loss = criterion(outputs, labels)
            
            losses.update(loss.item(), images.size(0))
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            pbar.set_postfix({'loss': losses.avg})
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics = {
        'val_loss': losses.avg,
        'val_acc': accuracy_score(all_labels, all_preds),
        'val_precision': precision_score(all_labels, all_preds, zero_division=0),
        'val_recall': recall_score(all_labels, all_preds, zero_division=0),
        'val_f1': f1_score(all_labels, all_preds, zero_division=0),
    }
    
    if len(np.unique(all_labels)) > 1:
        metrics['val_auc'] = roc_auc_score(all_labels, all_probs)
    else:
        metrics['val_auc'] = 0.0
    
    # Store for confusion matrix
    metrics['all_preds'] = all_preds
    metrics['all_labels'] = all_labels
    metrics['all_probs'] = all_probs
    
    return metrics


def plot_training_curves(history: dict, save_path: str) -> None:
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # AUC
    axes[1, 0].plot(epochs, history['train_auc'], 'b-', label='Train')
    axes[1, 0].plot(epochs, history['val_auc'], 'r-', label='Val')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].set_title('Training and Validation AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # F1 Score
    axes[1, 1].plot(epochs, history['train_f1'], 'b-', label='Train')
    axes[1, 1].plot(epochs, history['val_f1'], 'r-', label='Val')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('Training and Validation F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    save_figure(fig, save_path)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                          save_path: str, labels: list = ['Real', 'Fake']) -> None:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    save_figure(fig, save_path)


def plot_roc_curve(y_true: np.ndarray, y_probs: np.ndarray, 
                   save_path: str) -> None:
    """Plot and save ROC curve."""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, 'b-', label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'r--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc='lower right')
    ax.grid(True)
    
    save_figure(fig, save_path)


def train(config: dict) -> None:
    """
    Main training function.
    
    Args:
        config: Training configuration dictionary
    """
    # Setup
    set_seed(config.get('seed', 42))
    device = get_device()
    print(f"Using device: {device}")
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    reports_dir = Path(config['paths']['reports_dir'])
    create_directory(str(checkpoint_dir))
    create_directory(str(reports_dir / 'figures'))
    create_directory(str(reports_dir / 'metrics'))
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader = create_data_loaders(
        train_dir=config['data']['train_dir'],
        test_dir=config['data']['test_dir'],
        batch_size=config['training']['batch_size'],
        image_size=config['data']['image_size'],
        num_workers=config['data'].get('num_workers', 4),
        balanced_sampling=True
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        print("ERROR: No data found! Please add images to data/train and data/test directories.")
        return
    
    # Create model
    print("\nCreating model...")
    model = DeepfakeDetector(
        backbone=config['model']['backbone'],
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout'],
        freeze_backbone=config['model'].get('freeze_backbone', False)
    )
    model = model.to(device)
    
    print(f"Model: {config['model']['backbone']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=config['training']['epochs'],
        eta_min=config['training']['learning_rate'] / 100
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience'],
        mode='max'
    )
    
    # Metrics logger
    logger = MetricsLogger(str(reports_dir / 'metrics'), f'training_{timestamp}.csv')
    
    # Training loop
    print("\nStarting training...")
    history = {
        'train_loss': [], 'train_acc': [], 'train_auc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_auc': [], 'val_f1': []
    }
    
    best_val_auc = 0.0
    
    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, 
                                         optimizer, device, epoch)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        all_metrics = {**train_metrics, **{k: v for k, v in val_metrics.items() 
                                           if not k.startswith('all_')}}
        logger.log(all_metrics, epoch)
        
        # Update history
        for key in history:
            if key in all_metrics:
                history[key].append(all_metrics[key])
        
        # Print summary
        print(f"\nTrain - Loss: {train_metrics['train_loss']:.4f}, "
              f"Acc: {train_metrics['train_acc']:.4f}, "
              f"AUC: {train_metrics['train_auc']:.4f}, "
              f"F1: {train_metrics['train_f1']:.4f}")
        print(f"Val   - Loss: {val_metrics['val_loss']:.4f}, "
              f"Acc: {val_metrics['val_acc']:.4f}, "
              f"AUC: {val_metrics['val_auc']:.4f}, "
              f"F1: {val_metrics['val_f1']:.4f}")
        
        # Save best model
        if val_metrics['val_auc'] > best_val_auc:
            best_val_auc = val_metrics['val_auc']
            save_model(model, str(checkpoint_dir / 'best_model.pth'),
                      optimizer, epoch, val_metrics)
            print(f"  New best model saved! (AUC: {best_val_auc:.4f})")
        
        # Early stopping
        if early_stopping(val_metrics['val_auc']):
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    # Save final model
    save_model(model, str(checkpoint_dir / 'final_model.pth'),
               optimizer, epoch, val_metrics)
    
    # Plot training curves
    plot_training_curves(history, str(reports_dir / 'figures' / 'training_curves.png'))
    
    # Plot confusion matrix
    plot_confusion_matrix(
        val_metrics['all_labels'],
        val_metrics['all_preds'],
        str(reports_dir / 'figures' / 'confusion_matrix.png')
    )
    
    # Plot ROC curve
    if len(np.unique(val_metrics['all_labels'])) > 1:
        plot_roc_curve(
            val_metrics['all_labels'],
            val_metrics['all_probs'],
            str(reports_dir / 'figures' / 'roc_curve.png')
        )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print(f"Models saved to: {checkpoint_dir}")
    print(f"Reports saved to: {reports_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Train Deepfake Detector')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Override data directory')
    parser.add_argument('--quick_test', action='store_true',
                        help='Quick test with minimal data')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line args
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.data_dir is not None:
        config['data']['train_dir'] = os.path.join(args.data_dir, 'train')
        config['data']['test_dir'] = os.path.join(args.data_dir, 'test')
    
    if args.quick_test:
        config['training']['epochs'] = 2
        config['training']['batch_size'] = 4
    
    # Print config
    print("Configuration:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Train
    train(config)


if __name__ == "__main__":
    main()
