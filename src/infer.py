"""
Inference Module for Deepfake Detection

Provides functions for:
- Loading trained models (PyTorch, Keras, Classical ML)
- Preprocessing images for inference
- Making predictions
- Generating Grad-CAM heatmaps
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import base64
from io import BytesIO
from typing import Tuple, Optional, Dict, Any, Union
from pathlib import Path
import sys
import pickle
import importlib

# Add src to path FIRST
_src_path = str(Path(__file__).parent)
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# Import local modules dynamically to avoid auto-formatter issues
_module3 = importlib.import_module('module3_model')
_gradcam_module = importlib.import_module('gradcam')
DeepfakeDetector = _module3.DeepfakeDetector
GradCAM = _gradcam_module.GradCAM
heatmap_to_base64 = _gradcam_module.heatmap_to_base64


# Try to import Keras/TensorFlow (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    tf = None
    keras = None


# Default transforms for inference
INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model(model_path: str,
               device: Optional[torch.device] = None) -> Union[DeepfakeDetector, Any]:
    """
    Load a trained model from checkpoint.
    Supports PyTorch (.pth), Keras (.h5), and Classical ML (.pkl) models.

    Args:
        model_path: Path to model checkpoint
        device: Device to load model on (for PyTorch only)

    Returns:
        Loaded model
    """
    model_path = Path(model_path)

    # Determine model type by extension
    if model_path.suffix == '.pth':
        # PyTorch model
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(
            model_path, map_location=device, weights_only=False)
        backbone = checkpoint.get('backbone', 'resnet18')

        model = DeepfakeDetector(backbone=backbone, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        return model

    elif model_path.suffix == '.h5':
        # Keras model
        if not KERAS_AVAILABLE:
            raise ImportError(
                "TensorFlow/Keras not installed. Install with: pip install tensorflow")

        model = keras.models.load_model(str(model_path))
        return model

    elif model_path.suffix == '.pkl':
        # Classical ML model (sklearn)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model

    else:
        raise ValueError(
            f"Unsupported model format: {model_path.suffix}. Supported: .pth, .h5, .pkl")


def preprocess_image(image_path: str,
                     transform: Optional[transforms.Compose] = None) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Load and preprocess an image for inference.

    Args:
        image_path: Path to image file
        transform: Custom transform (uses default if None)

    Returns:
        Tuple of (tensor for model, original image as numpy array)
    """
    if transform is None:
        transform = INFERENCE_TRANSFORMS

    # Load with PIL for consistent color handling
    pil_image = Image.open(image_path).convert('RGB')

    # Also load with OpenCV for Grad-CAM overlay
    original_cv = cv2.imread(image_path)
    if original_cv is None:
        # Fallback: convert from PIL
        original_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Apply transforms
    tensor = transform(pil_image)

    return tensor, original_cv


def predict(model: Union[DeepfakeDetector, Any],
            image_path: str,
            device: Optional[torch.device] = None,
            generate_heatmap: bool = True) -> Dict[str, Any]:
    """
    Make a prediction on a single image.
    Supports PyTorch, Keras, and Classical ML models.

    Args:
        model: Trained model (PyTorch, Keras, or Classical ML)
        image_path: Path to image
        device: Device for inference (PyTorch only)
        generate_heatmap: Whether to generate Grad-CAM heatmap (PyTorch only)

    Returns:
        Dictionary with:
            - prob: Probability of being fake (0-1)
            - label: 'fake' or 'real'
            - confidence: Confidence percentage
            - heatmap: Numpy array of heatmap (if generate_heatmap=True and PyTorch)
            - heatmap_b64: Base64 encoded heatmap overlay (if generate_heatmap=True and PyTorch)
    """
    # Determine model type
    is_pytorch = isinstance(model, (DeepfakeDetector, nn.Module))
    is_keras = KERAS_AVAILABLE and isinstance(model, keras.Model)
    is_classical = hasattr(
        model, 'predict_proba') and not is_pytorch and not is_keras

    if is_pytorch:
        return _predict_pytorch(model, image_path, device, generate_heatmap)
    elif is_keras:
        return _predict_keras(model, image_path)
    elif is_classical:
        return _predict_classical(model, image_path)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


def _predict_pytorch(model: DeepfakeDetector,
                     image_path: str,
                     device: Optional[torch.device] = None,
                     generate_heatmap: bool = True) -> Dict[str, Any]:
    """PyTorch model prediction."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Preprocess image
    tensor, original = preprocess_image(image_path)
    tensor = tensor.unsqueeze(0).to(device)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).item()

    # Determine label
    label = 'fake' if prob > 0.5 else 'real'
    confidence = prob if prob > 0.5 else (1 - prob)

    result = {
        'prob': prob,
        'label': label,
        'confidence': confidence * 100,
        'image_path': image_path
    }

    # Generate heatmap if requested
    if generate_heatmap:
        # Need gradients for Grad-CAM
        tensor_with_grad = tensor.requires_grad_(True)

        gradcam = GradCAM(model)
        heatmap = gradcam(tensor_with_grad)

        # Generate overlay
        overlay = gradcam.generate_heatmap_overlay(tensor_with_grad, original)

        # Convert to base64
        heatmap_b64 = heatmap_to_base64(heatmap, original)

        result['heatmap'] = heatmap
        result['heatmap_b64'] = heatmap_b64
        result['overlay'] = overlay

    return result


def _predict_keras(model: Any, image_path: str) -> Dict[str, Any]:
    """Keras model prediction."""
    # Load and preprocess image for Keras
    # Try 112x112 first (expected by dfd-model.h5 which needs 12544 features)
    img = keras.preprocessing.image.load_img(
        image_path, target_size=(112, 112))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0, 1]

    # Make prediction
    pred = model.predict(img_array, verbose=0)[0][0]
    prob = float(pred)

    # Determine label
    label = 'fake' if prob > 0.5 else 'real'
    confidence = prob if prob > 0.5 else (1 - prob)

    return {
        'prob': prob,
        'label': label,
        'confidence': confidence * 100,
        'image_path': image_path
    }


def _predict_classical(model: Any, image_path: str) -> Dict[str, Any]:
    """Classical ML model prediction."""
    from module2_features import extract_all_features

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Extract features
    features = extract_all_features(img, ['hog', 'lbp', 'color', 'hu', 'edge'])
    features = features.reshape(1, -1)  # Reshape to (1, n_features)

    # Make prediction
    proba = model.predict_proba(features)[0]
    prob = float(proba[1])  # Probability of class 1 (fake)

    # Determine label
    label = 'fake' if prob > 0.5 else 'real'
    confidence = prob if prob > 0.5 else (1 - prob)

    return {
        'prob': prob,
        'label': label,
        'confidence': confidence * 100,
        'image_path': image_path
    }


def predict_batch(model: DeepfakeDetector,
                  image_paths: list,
                  device: Optional[torch.device] = None,
                  batch_size: int = 32) -> list:
    """
    Make predictions on multiple images.

    Args:
        model: Trained model
        image_paths: List of image paths
        device: Device for inference
        batch_size: Batch size for inference

    Returns:
        List of prediction dictionaries
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    results = []

    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        tensors = []
        valid_indices = []

        for j, path in enumerate(batch_paths):
            try:
                tensor, _ = preprocess_image(path)
                tensors.append(tensor)
                valid_indices.append(j)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                results.append({
                    'image_path': path,
                    'error': str(e)
                })

        if not tensors:
            continue

        # Stack and predict
        batch_tensor = torch.stack(tensors).to(device)

        with torch.no_grad():
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        # Handle single prediction case
        if probs.ndim == 0:
            probs = np.array([probs])

        # Store results
        for j, prob in zip(valid_indices, probs):
            path = batch_paths[j]
            label = 'fake' if prob > 0.5 else 'real'
            confidence = prob if prob > 0.5 else (1 - prob)

            results.append({
                'image_path': path,
                'prob': float(prob),
                'label': label,
                'confidence': float(confidence * 100)
            })

    return results


def save_heatmap(heatmap: np.ndarray,
                 original_image: np.ndarray,
                 save_path: str,
                 alpha: float = 0.5) -> None:
    """
    Save Grad-CAM heatmap overlay to file.

    Args:
        heatmap: Heatmap array (H, W)
        original_image: Original image (BGR)
        save_path: Output path
        alpha: Overlay transparency
    """
    # Resize heatmap
    h, w = original_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert to uint8 and apply colormap
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Blend
    overlay = cv2.addWeighted(
        original_image, 1 - alpha, heatmap_colored, alpha, 0)

    # Save
    cv2.imwrite(save_path, overlay)


class DeepfakePredictor:
    """
    Convenience class for making predictions.

    Loads model once and provides simple predict interface.
    """

    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize predictor.

        Args:
            model_path: Path to model checkpoint
            device: Device string ('cuda', 'cpu', or None for auto)
        """
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = load_model(model_path, self.device)
        self.gradcam = GradCAM(self.model)

    def __call__(self, image_path: str,
                 generate_heatmap: bool = True) -> Dict[str, Any]:
        """Make prediction on an image."""
        return predict(self.model, image_path, self.device, generate_heatmap)

    def predict_batch(self, image_paths: list,
                      batch_size: int = 32) -> list:
        """Make predictions on multiple images."""
        return predict_batch(self.model, image_paths, self.device, batch_size)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Deepfake Detection Inference')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to image')
    parser.add_argument('--save_heatmap', type=str, default=None,
                        help='Path to save heatmap overlay')

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}")
    model = load_model(args.model)

    # Make prediction
    print(f"Analyzing {args.image}")
    result = predict(model, args.image)

    print(f"\nResult:")
    print(f"  Label: {result['label']}")
    print(f"  Probability (fake): {result['prob']:.4f}")
    print(f"  Confidence: {result['confidence']:.1f}%")

    # Save heatmap if requested
    if args.save_heatmap and 'heatmap' in result:
        _, original = preprocess_image(args.image)
        save_heatmap(result['heatmap'], original, args.save_heatmap)
        print(f"\nHeatmap saved to {args.save_heatmap}")
