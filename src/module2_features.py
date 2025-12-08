"""
Module 2: Classical Feature Extraction and Baseline Classifier

This module provides:
- HOG (Histogram of Oriented Gradients) features
- LBP (Local Binary Patterns) features
- Color histogram features
- Hu moments
- SVM and RandomForest classifiers
"""

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
from typing import Tuple, List, Optional, Dict, Any
import pickle
import os
from tqdm import tqdm


# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def extract_hog(img: np.ndarray, 
                orientations: int = 9,
                pixels_per_cell: Tuple[int, int] = (8, 8),
                cells_per_block: Tuple[int, int] = (2, 2),
                resize_to: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    Extract Histogram of Oriented Gradients (HOG) features.
    
    HOG captures edge and gradient structure, useful for detecting
    manipulation artifacts at boundaries.
    
    Args:
        img: Input image (BGR or grayscale)
        orientations: Number of gradient orientations
        pixels_per_cell: Size of each cell
        cells_per_block: Number of cells per block
        resize_to: Resize image for consistent feature size
    
    Returns:
        HOG feature vector
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Resize for consistent feature dimensions
    gray = cv2.resize(gray, resize_to)
    
    # Extract HOG features
    features = hog(gray, 
                   orientations=orientations,
                   pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block,
                   block_norm='L2-Hys',
                   feature_vector=True)
    
    return features


def extract_lbp(img: np.ndarray,
                radius: int = 3,
                n_points: int = 24,
                method: str = 'uniform',
                resize_to: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    Extract Local Binary Pattern (LBP) features.
    
    LBP captures local texture patterns, effective for detecting
    inconsistent textures in manipulated regions.
    
    Args:
        img: Input image
        radius: Radius of circular LBP
        n_points: Number of points in LBP
        method: LBP method ('uniform', 'default', 'ror', 'var')
        resize_to: Resize image for consistent feature size
    
    Returns:
        LBP histogram feature vector
    """
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Resize
    gray = cv2.resize(gray, resize_to)
    
    # Compute LBP
    lbp = local_binary_pattern(gray, n_points, radius, method=method)
    
    # Compute histogram as features
    n_bins = n_points + 2 if method == 'uniform' else 256
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    return hist


def extract_color_histogram(img: np.ndarray,
                           bins_per_channel: int = 32,
                           color_space: str = 'hsv') -> np.ndarray:
    """
    Extract color histogram features.
    
    Color histograms capture overall color distribution, which may differ
    between real and synthesized images.
    
    Args:
        img: Input image (BGR)
        bins_per_channel: Number of bins per color channel
        color_space: 'hsv', 'bgr', or 'lab'
    
    Returns:
        Concatenated color histogram
    """
    # Convert color space
    if color_space == 'hsv':
        converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif color_space == 'lab':
        converted = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    else:
        converted = img
    
    # Compute histogram for each channel
    histograms = []
    for i in range(3):
        hist = cv2.calcHist([converted], [i], None, [bins_per_channel], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histograms.append(hist)
    
    return np.concatenate(histograms)


def extract_hu_moments(img: np.ndarray) -> np.ndarray:
    """
    Extract Hu moments (shape descriptors).
    
    Hu moments are invariant to translation, scale, and rotation,
    capturing overall shape characteristics.
    
    Args:
        img: Input image
    
    Returns:
        7 Hu moments (log-transformed)
    """
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Compute moments
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Log transform for better numerical stability
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    
    return hu_moments


def extract_edge_features(img: np.ndarray,
                          resize_to: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    Extract edge-based features using Canny + statistics.
    
    Edge features can reveal unnatural edge patterns in deepfakes.
    
    Args:
        img: Input image
        resize_to: Resize dimensions
    
    Returns:
        Edge statistics feature vector
    """
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Resize
    gray = cv2.resize(gray, resize_to)
    
    # Compute edges with different thresholds
    edges_low = cv2.Canny(gray, 50, 100)
    edges_high = cv2.Canny(gray, 100, 200)
    
    # Compute Sobel gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    
    # Compute statistics
    features = [
        np.mean(edges_low),
        np.std(edges_low),
        np.mean(edges_high),
        np.std(edges_high),
        np.mean(gradient_mag),
        np.std(gradient_mag),
        np.percentile(gradient_mag, 75),
        np.percentile(gradient_mag, 95),
    ]
    
    return np.array(features)


def extract_all_features(img: np.ndarray, 
                         feature_types: List[str] = None) -> np.ndarray:
    """
    Extract and concatenate multiple feature types.
    
    Args:
        img: Input image
        feature_types: List of features to extract. Options:
                      ['hog', 'lbp', 'color', 'hu', 'edge']
                      If None, uses all.
    
    Returns:
        Concatenated feature vector
    """
    if feature_types is None:
        feature_types = ['hog', 'lbp', 'color', 'hu', 'edge']
    
    features = []
    
    if 'hog' in feature_types:
        features.append(extract_hog(img))
    
    if 'lbp' in feature_types:
        features.append(extract_lbp(img))
    
    if 'color' in feature_types:
        features.append(extract_color_histogram(img))
    
    if 'hu' in feature_types:
        features.append(extract_hu_moments(img))
    
    if 'edge' in feature_types:
        features.append(extract_edge_features(img))
    
    return np.concatenate(features)


def get_feature_dimensions(feature_types: List[str] = None) -> Dict[str, int]:
    """
    Get expected dimensions for each feature type.
    
    Returns:
        Dictionary mapping feature type to dimension
    """
    # Based on default parameters
    dims = {
        'hog': 8100,  # For 128x128 with default params
        'lbp': 26,    # uniform LBP with 24 points
        'color': 96,  # 32 bins x 3 channels
        'hu': 7,      # 7 Hu moments
        'edge': 8,    # 8 edge statistics
    }
    
    if feature_types is None:
        return dims
    
    return {k: v for k, v in dims.items() if k in feature_types}


# ============================================================================
# CLASSIFIER
# ============================================================================

class ClassicalClassifier:
    """
    Classical machine learning classifier for deepfake detection.
    
    Supports SVM and RandomForest with optional PCA dimensionality reduction.
    """
    
    def __init__(self, 
                 classifier_type: str = 'svm',
                 use_pca: bool = True,
                 pca_components: int = 100,
                 random_state: int = 42):
        """
        Initialize classifier.
        
        Args:
            classifier_type: 'svm' or 'rf' (RandomForest)
            use_pca: Whether to apply PCA
            pca_components: Number of PCA components
            random_state: Random seed
        """
        self.classifier_type = classifier_type
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_components) if use_pca else None
        
        if classifier_type == 'svm':
            self.classifier = SVC(kernel='rbf', probability=True, 
                                  random_state=random_state, C=1.0, gamma='scale')
        elif classifier_type == 'rf':
            self.classifier = RandomForestClassifier(n_estimators=100,
                                                     random_state=random_state,
                                                     n_jobs=-1)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the classifier.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA if enabled
        if self.use_pca:
            n_components = min(self.pca_components, X_scaled.shape[0], X_scaled.shape[1])
            self.pca = PCA(n_components=n_components)
            X_reduced = self.pca.fit_transform(X_scaled)
        else:
            X_reduced = X_scaled
        
        # Fit classifier
        self.classifier.fit(X_reduced, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for samples.
        
        Args:
            X: Feature matrix
        
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted yet!")
        
        X_scaled = self.scaler.transform(X)
        if self.use_pca:
            X_reduced = self.pca.transform(X_scaled)
        else:
            X_reduced = X_scaled
        
        return self.classifier.predict(X_reduced)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
        
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted yet!")
        
        X_scaled = self.scaler.transform(X)
        if self.use_pca:
            X_reduced = self.pca.transform(X_scaled)
        else:
            X_reduced = X_scaled
        
        return self.classifier.predict_proba(X_reduced)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate classifier on test data.
        
        Args:
            X: Test features
            y: True labels
        
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'auc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0,
        }
        
        return metrics
    
    def get_confusion_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Get confusion matrix."""
        y_pred = self.predict(X)
        return confusion_matrix(y, y_pred)
    
    def get_roc_curve(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get ROC curve data."""
        y_proba = self.predict_proba(X)[:, 1]
        return roc_curve(y, y_proba)
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                       cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Labels
            cv: Number of folds
        
        Returns:
            Cross-validation scores
        """
        # Scale and reduce
        X_scaled = self.scaler.fit_transform(X)
        if self.use_pca:
            n_components = min(self.pca_components, X_scaled.shape[0], X_scaled.shape[1])
            pca = PCA(n_components=n_components)
            X_reduced = pca.fit_transform(X_scaled)
        else:
            X_reduced = X_scaled
        
        # Cross-validate
        scores = cross_val_score(self.classifier, X_reduced, y, cv=cv, scoring='accuracy')
        
        return {
            'cv_accuracy_mean': scores.mean(),
            'cv_accuracy_std': scores.std(),
        }
    
    def save(self, path: str) -> None:
        """Save classifier to file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'pca': self.pca,
                'classifier': self.classifier,
                'is_fitted': self.is_fitted,
                'classifier_type': self.classifier_type,
                'use_pca': self.use_pca,
            }, f)
    
    def load(self, path: str) -> None:
        """Load classifier from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.scaler = data['scaler']
        self.pca = data['pca']
        self.classifier = data['classifier']
        self.is_fitted = data['is_fitted']
        self.classifier_type = data['classifier_type']
        self.use_pca = data['use_pca']


# ============================================================================
# FEATURE EXTRACTION PIPELINE
# ============================================================================

def extract_features_from_dataset(image_paths: List[str],
                                  labels: List[int],
                                  feature_types: List[str] = None,
                                  show_progress: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from a dataset of images.
    
    Args:
        image_paths: List of image file paths
        labels: List of corresponding labels
        feature_types: Feature types to extract
        show_progress: Show progress bar
    
    Returns:
        Feature matrix (n_samples, n_features) and labels array
    """
    features_list = []
    valid_labels = []
    
    iterator = tqdm(zip(image_paths, labels), total=len(image_paths), 
                    desc="Extracting features") if show_progress else zip(image_paths, labels)
    
    for path, label in iterator:
        img = cv2.imread(path)
        if img is None:
            continue
        
        try:
            features = extract_all_features(img, feature_types)
            features_list.append(features)
            valid_labels.append(label)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
    
    return np.array(features_list), np.array(valid_labels)


def feature_ablation_study(X: np.ndarray, y: np.ndarray,
                           feature_info: Dict[str, Tuple[int, int]],
                           classifier_type: str = 'svm') -> Dict[str, Dict[str, float]]:
    """
    Perform ablation study by removing each feature type.
    
    Args:
        X: Full feature matrix
        y: Labels
        feature_info: Dict mapping feature name to (start_idx, end_idx)
        classifier_type: Type of classifier
    
    Returns:
        Dictionary of results for each ablation
    """
    results = {}
    
    # Full model
    clf_full = ClassicalClassifier(classifier_type=classifier_type, use_pca=True)
    results['all_features'] = clf_full.cross_validate(X, y)
    
    # Ablate each feature
    for feature_name, (start, end) in feature_info.items():
        # Create mask to exclude this feature
        mask = np.ones(X.shape[1], dtype=bool)
        mask[start:end] = False
        X_ablated = X[:, mask]
        
        clf = ClassicalClassifier(classifier_type=classifier_type, use_pca=True)
        results[f'without_{feature_name}'] = clf.cross_validate(X_ablated, y)
    
    return results


if __name__ == "__main__":
    # Quick demo with synthetic data
    print("Testing Module 2: Classical Features")
    
    # Create synthetic test image
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Test each feature extractor
    print("\nFeature dimensions:")
    hog_feat = extract_hog(test_img)
    print(f"  HOG: {hog_feat.shape}")
    
    lbp_feat = extract_lbp(test_img)
    print(f"  LBP: {lbp_feat.shape}")
    
    color_feat = extract_color_histogram(test_img)
    print(f"  Color histogram: {color_feat.shape}")
    
    hu_feat = extract_hu_moments(test_img)
    print(f"  Hu moments: {hu_feat.shape}")
    
    edge_feat = extract_edge_features(test_img)
    print(f"  Edge features: {edge_feat.shape}")
    
    all_feat = extract_all_features(test_img)
    print(f"  All features: {all_feat.shape}")
    
    # Test classifier with synthetic data
    print("\nTesting classifier with synthetic data...")
    n_samples = 100
    X = np.random.randn(n_samples, 500)
    y = np.random.randint(0, 2, n_samples)
    
    clf = ClassicalClassifier(classifier_type='svm', use_pca=True, pca_components=50)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit and evaluate
    clf.fit(X_train, y_train)
    metrics = clf.evaluate(X_test, y_test)
    
    print(f"\nTest metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    print("\nModule 2 loaded successfully!")
