"""
Model Comparison Framework for Deepfake Detection
Compares classical ML (SVM, RF) vs. deep learning (ResNet18) with statistical analysis
"""

import os
import sys
import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cv2

# Statistical tests
from scipy import stats
from scipy.stats import ttest_rel, ttest_ind, f_oneway, friedmanchisquare
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

import torch
import torch.nn as nn
from torchvision import transforms

# Import local modules
from module2_features import extract_all_features, ClassicalClassifier
from module3_model import DeepfakeDetector
from utils import get_image_files


class ModelComparison:
    """
    Comprehensive model comparison framework with statistical analysis
    """

    def __init__(self, data_dir, results_dir='reports/comparison', device='cuda'):
        """
        Initialize comparison framework

        Args:
            data_dir: Path to dataset directory
            results_dir: Directory to save comparison results
            device: Device for deep learning ('cuda' or 'cpu')
        """
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.inference_times = {}

    def load_dataset(self, split='test', max_samples=None):
        """
        Load dataset for evaluation

        Args:
            split: 'train' or 'test'
            max_samples: Maximum samples per class (None for all)

        Returns:
            images: List of BGR images
            labels: Array of labels (0=real, 1=fake)
        """
        print(f"\nLoading {split} dataset...")

        split_dir = self.data_dir / split.capitalize()
        images = []
        labels = []

        for label, folder_name in [(0, 'Real'), (1, 'Fake')]:
            folder_path = split_dir / folder_name
            if not folder_path.exists():
                folder_path = split_dir / folder_name.lower()

            if not folder_path.exists():
                print(
                    f"Warning: {folder_name} folder not found at {folder_path}")
                continue

            files = get_image_files(str(folder_path))
            if max_samples:
                np.random.seed(42)
                np.random.shuffle(files)
                files = files[:max_samples]

            for f in tqdm(files, desc=f"Loading {folder_name}"):
                img = cv2.imread(f)
                if img is not None:
                    images.append(img)
                    labels.append(label)

        labels = np.array(labels)
        print(
            f"Loaded {len(images)} images: {np.sum(labels == 0)} real, {np.sum(labels == 1)} fake")

        return images, labels

    def load_classical_model(self, model_path, model_name='Classical-RF'):
        """
        Load trained classical classifier

        Args:
            model_path: Path to saved model (.pkl)
            model_name: Name for this model
        """
        print(f"\nLoading classical model: {model_name}")

        with open(model_path, 'rb') as f:
            clf = pickle.load(f)

        self.models[model_name] = {
            'type': 'classical',
            'model': clf,
            'feature_types': getattr(clf, 'feature_types', ['hog', 'lbp', 'color'])
        }

        print(f"  Classifier: {clf.classifier_type.upper()}")
        print(f"  Features: {self.models[model_name]['feature_types']}")
        print(
            f"  PCA: {clf.use_pca} ({clf.pca_components if clf.use_pca else 'N/A'} components)")

    def load_deep_model(self, model_path, model_name='ResNet18'):
        """
        Load trained deep learning model

        Args:
            model_path: Path to saved model (.pth)
            model_name: Name for this model
        """
        print(f"\nLoading deep learning model: {model_name}")

        model = DeepfakeDetector()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()

        self.models[model_name] = {
            'type': 'deep',
            'model': model,
            'transform': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ])
        }

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

    def extract_classical_features(self, images, feature_types):
        """Extract classical features from images"""
        print(f"Extracting features: {feature_types}")
        features = []
        for img in tqdm(images, desc="Feature extraction"):
            feat = extract_all_features(img, feature_types)
            features.append(feat)
        return np.array(features)

    def predict_classical(self, model_info, images):
        """
        Generate predictions with classical model

        Returns:
            probs: Probability of fake (0-1)
            preds: Binary predictions
            inference_time: Average inference time per image (ms)
        """
        clf = model_info['model']
        feature_types = model_info['feature_types']

        # Extract features
        X = self.extract_classical_features(images, feature_types)

        # Time inference
        import time
        start = time.time()
        probs = clf.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        inference_time = (time.time() - start) / \
            len(images) * 1000  # ms per image

        return probs, preds, inference_time

    def predict_deep(self, model_info, images):
        """
        Generate predictions with deep learning model

        Returns:
            probs: Probability of fake (0-1)
            preds: Binary predictions
            inference_time: Average inference time per image (ms)
        """
        model = model_info['model']
        transform = model_info['transform']

        probs = []
        preds = []

        import time
        total_time = 0

        with torch.no_grad():
            for img in tqdm(images, desc="Deep inference"):
                # Preprocess
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = transform(img_rgb).unsqueeze(0).to(self.device)

                # Inference
                start = time.time()
                output = model(img_tensor)
                total_time += (time.time() - start)

                prob = output.item()
                probs.append(prob)
                preds.append(1 if prob >= 0.5 else 0)

        inference_time = total_time / len(images) * 1000  # ms per image

        return np.array(probs), np.array(preds), inference_time

    def evaluate_model(self, model_name, images, labels):
        """
        Evaluate a single model and store results

        Args:
            model_name: Name of model to evaluate
            images: List of images
            labels: Ground truth labels
        """
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        model_info = self.models[model_name]

        # Generate predictions
        if model_info['type'] == 'classical':
            probs, preds, inference_time = self.predict_classical(
                model_info, images)
        else:
            probs, preds, inference_time = self.predict_deep(
                model_info, images)

        # Store predictions
        self.predictions[model_name] = {
            'probs': probs,
            'preds': preds
        }

        self.inference_times[model_name] = inference_time

        # Calculate metrics
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        auc = roc_auc_score(labels, probs)
        ap = average_precision_score(labels, probs)

        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        self.metrics[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'ap': ap,
            'specificity': specificity,
            'confusion_matrix': cm,
            'inference_time': inference_time
        }

        # Print results
        print(f"\nMetrics:")
        print(f"  Accuracy:    {accuracy:.4f}")
        print(f"  Precision:   {precision:.4f}")
        print(f"  Recall:      {recall:.4f}")
        print(f"  F1 Score:    {f1:.4f}")
        print(f"  AUC-ROC:     {auc:.4f}")
        print(f"  AP:          {ap:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {tn:5d}  FP: {fp:5d}")
        print(f"  FN: {fn:5d}  TP: {tp:5d}")
        print(f"\nInference Time: {inference_time:.2f} ms/image")

    def statistical_comparison(self, labels):
        """
        Perform statistical comparison between models

        Args:
            labels: Ground truth labels
        """
        print(f"\n{'='*60}")
        print("STATISTICAL COMPARISON")
        print(f"{'='*60}")

        model_names = list(self.predictions.keys())

        if len(model_names) < 2:
            print("Need at least 2 models for comparison")
            return

        # Pairwise comparisons
        print("\n### Pairwise McNemar's Test (Accuracy Differences) ###")
        from statsmodels.stats.contingency_tables import mcnemar

        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                pred1 = self.predictions[model1]['preds']
                pred2 = self.predictions[model2]['preds']

                # Create contingency table
                correct1 = (pred1 == labels)
                correct2 = (pred2 == labels)

                # 2x2 table: [both_correct, model1_only, model2_only, both_wrong]
                table = np.array([
                    [np.sum(correct1 & correct2),
                     np.sum(correct1 & ~correct2)],
                    [np.sum(~correct1 & correct2),
                     np.sum(~correct1 & ~correct2)]
                ])

                result = mcnemar(table, exact=False, correction=True)

                acc1 = self.metrics[model1]['accuracy']
                acc2 = self.metrics[model2]['accuracy']
                diff = acc1 - acc2

                print(f"\n{model1} vs {model2}:")
                print(
                    f"  Accuracy: {acc1:.4f} vs {acc2:.4f} (Δ = {diff:+.4f})")
                print(
                    f"  McNemar χ²: {result.statistic:.4f}, p-value: {result.pvalue:.4f}")

                if result.pvalue < 0.001:
                    sig = "***"
                elif result.pvalue < 0.01:
                    sig = "**"
                elif result.pvalue < 0.05:
                    sig = "*"
                else:
                    sig = "ns"
                print(
                    f"  Significance: {sig} ({'significant' if result.pvalue < 0.05 else 'not significant'})")

        # Effect sizes (Cohen's d for AUC differences)
        print("\n### Effect Sizes (Cohen's d) ###")

        # Bootstrap for confidence intervals
        n_bootstrap = 1000
        np.random.seed(42)

        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                auc1 = self.metrics[model1]['auc']
                auc2 = self.metrics[model2]['auc']

                # Bootstrap to estimate variance
                auc1_boots = []
                auc2_boots = []

                for _ in range(n_bootstrap):
                    indices = np.random.choice(
                        len(labels), len(labels), replace=True)
                    boot_labels = labels[indices]
                    boot_probs1 = self.predictions[model1]['probs'][indices]
                    boot_probs2 = self.predictions[model2]['probs'][indices]

                    auc1_boots.append(roc_auc_score(boot_labels, boot_probs1))
                    auc2_boots.append(roc_auc_score(boot_labels, boot_probs2))

                pooled_std = np.sqrt(
                    (np.std(auc1_boots)**2 + np.std(auc2_boots)**2) / 2)
                cohens_d = (auc1 - auc2) / pooled_std if pooled_std > 0 else 0

                print(f"\n{model1} vs {model2}:")
                print(f"  AUC: {auc1:.4f} vs {auc2:.4f}")
                print(f"  Cohen's d: {cohens_d:.4f}", end="")

                if abs(cohens_d) >= 0.8:
                    effect = "large"
                elif abs(cohens_d) >= 0.5:
                    effect = "medium"
                elif abs(cohens_d) >= 0.2:
                    effect = "small"
                else:
                    effect = "negligible"
                print(f" ({effect} effect)")

    def create_comparison_table(self):
        """Create comprehensive comparison table"""
        print(f"\n{'='*60}")
        print("COMPREHENSIVE COMPARISON TABLE")
        print(f"{'='*60}\n")

        rows = []
        for model_name, metrics in self.metrics.items():
            model_type = self.models[model_name]['type']

            # Count parameters
            if model_type == 'deep':
                params = sum(p.numel()
                             for p in self.models[model_name]['model'].parameters())
                params_str = f"{params/1e6:.1f}M"
            else:
                clf = self.models[model_name]['model']
                params = clf.pca_components if clf.use_pca else 'N/A'
                params_str = str(params)

            rows.append({
                'Model': model_name,
                'Type': model_type.capitalize(),
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1': f"{metrics['f1']:.4f}",
                'AUC': f"{metrics['auc']:.4f}",
                'AP': f"{metrics['ap']:.4f}",
                'Params': params_str,
                'Inference (ms)': f"{metrics['inference_time']:.2f}"
            })

        df = pd.DataFrame(rows)
        print(df.to_string(index=False))

        # Save to CSV
        csv_path = self.results_dir / 'model_comparison.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nTable saved to: {csv_path}")

        return df

    def plot_comparison(self, labels):
        """Create comprehensive comparison visualizations"""
        print(f"\nGenerating comparison plots...")

        n_models = len(self.metrics)
        model_names = list(self.metrics.keys())

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Metrics Comparison Bar Chart
        ax1 = fig.add_subplot(gs[0, :2])
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        x = np.arange(n_models)
        width = 0.15

        for i, metric in enumerate(metrics_to_plot):
            values = [self.metrics[name][metric] for name in model_names]
            ax1.bar(x + i*width, values, width, label=metric.upper())

        ax1.set_xlabel('Model', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Performance Metrics Comparison',
                      fontsize=14, fontweight='bold')
        ax1.set_xticks(x + width * 2)
        ax1.set_xticklabels(model_names)
        ax1.legend(loc='lower right')
        ax1.set_ylim(0, 1.0)
        ax1.grid(axis='y', alpha=0.3)

        # 2. Inference Time Comparison
        ax2 = fig.add_subplot(gs[0, 2])
        inference_times = [self.metrics[name]['inference_time']
                           for name in model_names]
        colors = sns.color_palette('husl', n_models)
        bars = ax2.barh(model_names, inference_times, color=colors)
        ax2.set_xlabel('Inference Time (ms)', fontsize=12)
        ax2.set_title('Inference Speed', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, inference_times):
            ax2.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}',
                     va='center', fontsize=10)

        # 3. ROC Curves
        ax3 = fig.add_subplot(gs[1, 0])
        for name, color in zip(model_names, colors):
            probs = self.predictions[name]['probs']
            fpr, tpr, _ = roc_curve(labels, probs)
            auc = self.metrics[name]['auc']
            ax3.plot(fpr, tpr, linewidth=2,
                     label=f'{name} (AUC={auc:.3f})', color=color)

        ax3.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
        ax3.set_xlabel('False Positive Rate', fontsize=12)
        ax3.set_ylabel('True Positive Rate', fontsize=12)
        ax3.set_title('ROC Curves', fontsize=14, fontweight='bold')
        ax3.legend(loc='lower right')
        ax3.grid(alpha=0.3)

        # 4. Precision-Recall Curves
        ax4 = fig.add_subplot(gs[1, 1])
        for name, color in zip(model_names, colors):
            probs = self.predictions[name]['probs']
            precision, recall, _ = precision_recall_curve(labels, probs)
            ap = self.metrics[name]['ap']
            ax4.plot(recall, precision, linewidth=2,
                     label=f'{name} (AP={ap:.3f})', color=color)

        ax4.set_xlabel('Recall', fontsize=12)
        ax4.set_ylabel('Precision', fontsize=12)
        ax4.set_title('Precision-Recall Curves',
                      fontsize=14, fontweight='bold')
        ax4.legend(loc='lower left')
        ax4.grid(alpha=0.3)

        # 5. Confusion Matrices
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')

        # Create mini confusion matrices
        for i, name in enumerate(model_names):
            cm = self.metrics[name]['confusion_matrix']

            # Normalize
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            # Position
            left = 0.05
            bottom = 0.9 - i * 0.35
            width = 0.9
            height = 0.25

            ax_cm = fig.add_axes(
                [left + 0.67, bottom, width * 0.08, height * 0.8])
            sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
                        ax=ax_cm, square=True)
            ax_cm.set_title(name, fontsize=10, fontweight='bold')
            ax_cm.set_ylabel('True', fontsize=9)
            ax_cm.set_xlabel('Predicted', fontsize=9)

        # 6. Accuracy by Model
        ax6 = fig.add_subplot(gs[2, 0])
        accuracies = [self.metrics[name]['accuracy'] for name in model_names]
        bars = ax6.bar(model_names, accuracies, color=colors,
                       alpha=0.7, edgecolor='black')
        ax6.set_ylabel('Accuracy', fontsize=12)
        ax6.set_title('Overall Accuracy', fontsize=14, fontweight='bold')
        ax6.set_ylim(0.5, 1.0)
        ax6.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, accuracies):
            ax6.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

        # 7. Precision vs Recall Trade-off
        ax7 = fig.add_subplot(gs[2, 1])
        precisions = [self.metrics[name]['precision'] for name in model_names]
        recalls = [self.metrics[name]['recall'] for name in model_names]

        for i, (name, color) in enumerate(zip(model_names, colors)):
            ax7.scatter(recalls[i], precisions[i], s=200, color=color,
                        alpha=0.7, edgecolor='black', linewidth=2, label=name)
            ax7.annotate(name, (recalls[i], precisions[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=9)

        ax7.set_xlabel('Recall', fontsize=12)
        ax7.set_ylabel('Precision', fontsize=12)
        ax7.set_title('Precision-Recall Trade-off',
                      fontsize=14, fontweight='bold')
        ax7.grid(alpha=0.3)
        ax7.set_xlim(0.5, 1.0)
        ax7.set_ylim(0.5, 1.0)

        # 8. F1 Score Comparison
        ax8 = fig.add_subplot(gs[2, 2])
        f1_scores = [self.metrics[name]['f1'] for name in model_names]
        ax8.barh(model_names, f1_scores, color=colors,
                 alpha=0.7, edgecolor='black')
        ax8.set_xlabel('F1 Score', fontsize=12)
        ax8.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
        ax8.set_xlim(0.5, 1.0)
        ax8.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (name, val) in enumerate(zip(model_names, f1_scores)):
            ax8.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=10)

        plt.suptitle('Comprehensive Model Comparison',
                     fontsize=16, fontweight='bold', y=0.995)

        # Save figure
        plot_path = self.results_dir / 'model_comparison.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {plot_path}")

        plt.show()

    def save_results(self):
        """Save all results to JSON"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'statistical_tests': {}
        }

        for name, metrics in self.metrics.items():
            results['models'][name] = {
                'type': self.models[name]['type'],
                'metrics': {
                    'accuracy': float(metrics['accuracy']),
                    'precision': float(metrics['precision']),
                    'recall': float(metrics['recall']),
                    'f1': float(metrics['f1']),
                    'auc': float(metrics['auc']),
                    'ap': float(metrics['ap']),
                    'specificity': float(metrics['specificity']),
                    'inference_time_ms': float(metrics['inference_time'])
                },
                'confusion_matrix': metrics['confusion_matrix'].tolist()
            }

        json_path = self.results_dir / 'comparison_results.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare deepfake detection models')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to dataset directory')
    parser.add_argument('--classical_model', type=str, default='models/classical_rf.pkl',
                        help='Path to classical model (.pkl)')
    parser.add_argument('--deep_model', type=str, default='models/best_model.pth',
                        help='Path to deep learning model (.pth)')
    parser.add_argument('--svm_model', type=str, default=None,
                        help='Path to SVM model (.pkl, optional)')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='Max samples per class for evaluation')
    parser.add_argument('--results_dir', type=str, default='reports/comparison',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for deep learning (cuda/cpu)')

    args = parser.parse_args()

    print("="*60)
    print("MODEL COMPARISON FRAMEWORK")
    print("="*60)

    # Initialize comparison
    comparison = ModelComparison(args.data_dir, args.results_dir, args.device)

    # Load models
    models_loaded = 0

    if os.path.exists(args.classical_model):
        comparison.load_classical_model(args.classical_model, 'RandomForest')
        models_loaded += 1
    else:
        print(f"Warning: Classical model not found at {args.classical_model}")

    if args.svm_model and os.path.exists(args.svm_model):
        comparison.load_classical_model(args.svm_model, 'SVM')
        models_loaded += 1

    if os.path.exists(args.deep_model):
        comparison.load_deep_model(args.deep_model, 'ResNet18')
        models_loaded += 1
    else:
        print(f"Warning: Deep model not found at {args.deep_model}")

    if models_loaded < 2:
        print("\nError: Need at least 2 models for comparison!")
        print("Train models first using train_classical.py and train.py")
        return

    # Load test dataset
    images, labels = comparison.load_dataset(
        split='test', max_samples=args.max_samples)

    # Evaluate all models
    for model_name in comparison.models.keys():
        comparison.evaluate_model(model_name, images, labels)

    # Statistical comparison
    comparison.statistical_comparison(labels)

    # Create comparison table
    comparison.create_comparison_table()

    # Generate plots
    comparison.plot_comparison(labels)

    # Save results
    comparison.save_results()

    print("\n" + "="*60)
    print("COMPARISON COMPLETE!")
    print("="*60)
    print(f"Results saved to: {args.results_dir}")


if __name__ == '__main__':
    main()
