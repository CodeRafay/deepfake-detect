"""
Inference Module for Deepfake Detection

Provides functions for:
- Loading trained models
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
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from module3_model import DeepfakeDetector
from gradcam import GradCAM, heatmap_to_base64


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
               device: Optional[torch.device] = None) -> DeepfakeDetector:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use weights_only=False for our trusted checkpoints (PyTorch 2.6+ compatibility)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get backbone from checkpoint or use default
    backbone = checkpoint.get('backbone', 'resnet18')
    
    # Create model
    model = DeepfakeDetector(backbone=backbone, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


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


def predict(model: DeepfakeDetector,
            image_path: str,
            device: Optional[torch.device] = None,
            generate_heatmap: bool = True) -> Dict[str, Any]:
    """
    Make a prediction on a single image.
    
    Args:
        model: Trained DeepfakeDetector model
        image_path: Path to image
        device: Device for inference
        generate_heatmap: Whether to generate Grad-CAM heatmap
    
    Returns:
        Dictionary with:
            - prob: Probability of being fake (0-1)
            - label: 'fake' or 'real'
            - confidence: Confidence percentage
            - heatmap: Numpy array of heatmap (if generate_heatmap=True)
            - heatmap_b64: Base64 encoded heatmap overlay (if generate_heatmap=True)
    """
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
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
    
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
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    
    parser = argparse.ArgumentParser(description='Deepfake Detection Inference')
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
