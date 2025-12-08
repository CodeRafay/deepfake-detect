"""
Training Script for Classical ML Baseline (Module 2)

Trains SVM/RandomForest on HOG, LBP, and other handcrafted features.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from module2_features import (
    extract_all_features, extract_features_from_dataset,
    ClassicalClassifier, get_feature_dimensions
)
from utils import set_seed, get_image_files, create_directory, save_figure


def load_dataset(data_dir: str):
    """Load image paths and labels from dataset directory."""
    data_path = Path(data_dir)
    
    image_paths = []
    labels = []
    
    # Load real images (label = 0)
    real_dir = data_path / 'real'
    if real_dir.exists():
        real_images = get_image_files(str(real_dir))
        image_paths.extend(real_images)
        labels.extend([0] * len(real_images))
        print(f"Found {len(real_images)} real images")
    
    # Load fake images (label = 1)
    fake_dir = data_path / 'fake'
    if fake_dir.exists():
        fake_images = get_image_files(str(fake_dir))
        image_paths.extend(fake_images)
        labels.extend([1] * len(fake_images))
        print(f"Found {len(fake_images)} fake images")
    
    return image_paths, labels


def plot_confusion_matrix(y_true, y_pred, save_path, labels=['Real', 'Fake']):
    """Plot and save confusion matrix."""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Classical Baseline - Confusion Matrix')
    
    save_figure(fig, save_path)


def plot_roc_curve(y_true, y_probs, save_path):
    """Plot and save ROC curve."""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, 'b-', label=f'Classical Baseline (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'r--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Classical Baseline')
    ax.legend(loc='lower right')
    ax.grid(True)
    
    save_figure(fig, save_path)


def train_classical_baseline(args):
    """Train classical baseline classifier."""
    set_seed(args.seed)
    
    # Create output directories
    reports_dir = Path(args.reports_dir)
    create_directory(str(reports_dir / 'figures'))
    create_directory(str(reports_dir / 'metrics'))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load training data
    print("\n" + "="*60)
    print("Loading training data...")
    print("="*60)
    
    train_paths, train_labels = load_dataset(args.train_dir)
    
    if len(train_paths) == 0:
        print("ERROR: No training data found!")
        print(f"Please add images to: {args.train_dir}/real/ and {args.train_dir}/fake/")
        return
    
    # Extract features
    print("\n" + "="*60)
    print("Extracting features...")
    print("="*60)
    
    feature_types = args.features.split(',') if args.features else None
    X_train, y_train = extract_features_from_dataset(
        train_paths, train_labels, feature_types
    )
    
    print(f"Feature matrix shape: {X_train.shape}")
    
    # Load test data
    print("\n" + "="*60)
    print("Loading test data...")
    print("="*60)
    
    test_paths, test_labels = load_dataset(args.test_dir)
    
    if len(test_paths) > 0:
        X_test, y_test = extract_features_from_dataset(
            test_paths, test_labels, feature_types
        )
    else:
        print("No test data found, using train/test split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=args.seed, stratify=y_train
        )
    
    print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")
    
    # Train classifier
    print("\n" + "="*60)
    print(f"Training {args.classifier.upper()} classifier...")
    print("="*60)
    
    clf = ClassicalClassifier(
        classifier_type=args.classifier,
        use_pca=args.use_pca,
        pca_components=args.pca_components,
        random_state=args.seed
    )
    
    # Cross-validation
    print("\nPerforming cross-validation...")
    cv_results = clf.cross_validate(X_train, y_train, cv=5)
    print(f"CV Accuracy: {cv_results['cv_accuracy_mean']:.4f} ± {cv_results['cv_accuracy_std']:.4f}")
    
    # Train on full training set
    print("\nTraining on full training set...")
    clf.fit(X_train, y_train)
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    
    metrics = clf.evaluate(X_test, y_test)
    
    print(f"\nTest Set Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    
    # Save metrics to file
    metrics_path = reports_dir / 'metrics' / f'classical_baseline_{timestamp}.txt'
    with open(metrics_path, 'w') as f:
        f.write("Classical Baseline Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Classifier: {args.classifier.upper()}\n")
        f.write(f"Features: {args.features or 'all'}\n")
        f.write(f"PCA: {args.use_pca} (components: {args.pca_components})\n")
        f.write(f"Train samples: {len(y_train)}\n")
        f.write(f"Test samples: {len(y_test)}\n\n")
        f.write("Cross-Validation:\n")
        f.write(f"  Accuracy: {cv_results['cv_accuracy_mean']:.4f} ± {cv_results['cv_accuracy_std']:.4f}\n\n")
        f.write("Test Set Metrics:\n")
        for name, value in metrics.items():
            f.write(f"  {name}: {value:.4f}\n")
    
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Plot confusion matrix
    y_pred = clf.predict(X_test)
    cm_path = reports_dir / 'figures' / f'classical_confusion_matrix_{timestamp}.png'
    plot_confusion_matrix(y_test, y_pred, str(cm_path))
    print(f"Confusion matrix saved to: {cm_path}")
    
    # Plot ROC curve
    if len(np.unique(y_test)) > 1:
        y_probs = clf.predict_proba(X_test)[:, 1]
        roc_path = reports_dir / 'figures' / f'classical_roc_curve_{timestamp}.png'
        plot_roc_curve(y_test, y_probs, str(roc_path))
        print(f"ROC curve saved to: {roc_path}")
    
    # Save classifier
    model_path = Path(args.model_dir) / 'classical_baseline.pkl'
    create_directory(str(Path(args.model_dir)))
    clf.save(str(model_path))
    print(f"\nModel saved to: {model_path}")
    
    print("\n" + "="*60)
    print("Classical baseline training complete!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Train Classical Baseline')
    parser.add_argument('--train_dir', type=str, default='data/train',
                        help='Training data directory')
    parser.add_argument('--test_dir', type=str, default='data/test',
                        help='Test data directory')
    parser.add_argument('--classifier', type=str, default='svm',
                        choices=['svm', 'rf'],
                        help='Classifier type (svm or rf)')
    parser.add_argument('--features', type=str, default=None,
                        help='Comma-separated feature types (hog,lbp,color,hu,edge)')
    parser.add_argument('--use_pca', action='store_true',
                        help='Use PCA for dimensionality reduction')
    parser.add_argument('--pca_components', type=int, default=100,
                        help='Number of PCA components')
    parser.add_argument('--reports_dir', type=str, default='reports',
                        help='Reports output directory')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Model output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Use PCA by default for classical features (they're high-dimensional)
    args.use_pca = True
    
    train_classical_baseline(args)


if __name__ == "__main__":
    main()
