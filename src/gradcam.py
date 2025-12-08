"""
Grad-CAM Implementation for Deepfake Detection Explainability

Provides visual explanations of which image regions the model
focuses on when making predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional, List
from PIL import Image
import base64
from io import BytesIO


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Produces heatmaps showing which regions of an image contributed
    most to the model's prediction.
    """
    
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model (DeepfakeDetector or similar)
            target_layer: Layer to compute Grad-CAM for. If None,
                         automatically selects last conv layer.
        """
        self.model = model
        self.model.eval()
        
        # Find target layer if not specified
        if target_layer is None:
            target_layer = self._find_target_layer()
        
        self.target_layer = target_layer
        
        # Storage for gradients and activations
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _find_target_layer(self) -> nn.Module:
        """Find the last convolutional layer in the model."""
        target_layer = None
        
        # For ResNet models
        if hasattr(self.model, 'backbone'):
            backbone = self.model.backbone
            if hasattr(backbone, 'layer4'):
                target_layer = backbone.layer4[-1]
            elif hasattr(backbone, 'features'):
                # EfficientNet
                for layer in reversed(list(backbone.features.children())):
                    if isinstance(layer, nn.Conv2d) or hasattr(layer, 'conv'):
                        target_layer = layer
                        break
        
        if target_layer is None:
            # Fallback: find any conv layer
            for module in reversed(list(self.model.modules())):
                if isinstance(module, nn.Conv2d):
                    target_layer = module
                    break
        
        if target_layer is None:
            raise ValueError("Could not find target layer for Grad-CAM")
        
        return target_layer
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def __call__(self, input_tensor: torch.Tensor, 
                 target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class for gradient. If None, uses
                         predicted class.
        
        Returns:
            Heatmap as numpy array (H, W), normalized to [0, 1]
        """
        # Ensure model is in eval mode with gradients enabled
        self.model.eval()
        
        # Enable gradients for input
        input_tensor = input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Binary classification - use the single output
        if output.shape[-1] == 1 or len(output.shape) == 1:
            target = output.squeeze()
        else:
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            target = output[:, target_class]
        
        # Backward pass
        self.model.zero_grad()
        target.backward(retain_graph=True)
        
        # Compute Grad-CAM
        gradients = self.gradients  # (B, C, H, W)
        activations = self.activations  # (B, C, H, W)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        
        # Weighted combination of activations
        cam = torch.sum(weights * activations, dim=1)  # (B, H, W)
        
        # ReLU to keep positive contributions
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def generate_heatmap_overlay(self, 
                                 input_tensor: torch.Tensor,
                                 original_image: np.ndarray,
                                 target_class: Optional[int] = None,
                                 alpha: float = 0.5,
                                 colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Generate Grad-CAM heatmap overlaid on original image.
        
        Args:
            input_tensor: Preprocessed input tensor
            original_image: Original image (BGR, uint8)
            target_class: Target class for gradient
            alpha: Overlay transparency
            colormap: OpenCV colormap
        
        Returns:
            Overlaid image (BGR, uint8)
        """
        # Generate heatmap
        cam = self(input_tensor, target_class)
        
        # Resize to original image size
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Convert to uint8
        cam_uint8 = (cam_resized * 255).astype(np.uint8)
        
        # Apply colormap
        cam_colored = cv2.applyColorMap(cam_uint8, colormap)
        
        # Overlay
        overlay = cv2.addWeighted(original_image, 1 - alpha, cam_colored, alpha, 0)
        
        return overlay


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ for improved localization.
    
    Uses second-order gradients for better weighting.
    """
    
    def __call__(self, input_tensor: torch.Tensor,
                 target_class: Optional[int] = None) -> np.ndarray:
        """Generate Grad-CAM++ heatmap."""
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        if output.shape[-1] == 1 or len(output.shape) == 1:
            target = output.squeeze()
        else:
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            target = output[:, target_class]
        
        # First backward
        self.model.zero_grad()
        target.backward(retain_graph=True)
        
        gradients = self.gradients
        activations = self.activations
        
        # Grad-CAM++ weights
        grad_2 = gradients ** 2
        grad_3 = grad_2 * gradients
        
        sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
        
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-7
        alpha = alpha_num / alpha_denom
        
        weights = alpha * F.relu(gradients)
        weights = torch.sum(weights, dim=(2, 3), keepdim=True)
        
        cam = torch.sum(weights * activations, dim=1)
        cam = F.relu(cam)
        
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam


def compute_gradcam(model: nn.Module,
                    img_tensor: torch.Tensor,
                    original_image: Optional[np.ndarray] = None,
                    use_plus_plus: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convenience function to compute Grad-CAM.
    
    Args:
        model: PyTorch model
        img_tensor: Preprocessed image tensor (1, C, H, W)
        original_image: Original image for overlay (optional)
        use_plus_plus: Use Grad-CAM++ instead of Grad-CAM
    
    Returns:
        Tuple of (heatmap, overlay or None)
    """
    if use_plus_plus:
        gradcam = GradCAMPlusPlus(model)
    else:
        gradcam = GradCAM(model)
    
    heatmap = gradcam(img_tensor)
    
    overlay = None
    if original_image is not None:
        overlay = gradcam.generate_heatmap_overlay(img_tensor, original_image)
    
    return heatmap, overlay


def heatmap_to_base64(heatmap: np.ndarray, 
                      original_image: Optional[np.ndarray] = None,
                      alpha: float = 0.5) -> str:
    """
    Convert heatmap to base64 encoded image.
    
    Args:
        heatmap: Heatmap array (H, W) in [0, 1]
        original_image: Original image to overlay on
        alpha: Overlay transparency
    
    Returns:
        Base64 encoded PNG image
    """
    # If we have original image, create overlay
    if original_image is not None:
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        result = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    else:
        # Just the heatmap
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        result_rgb = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        result_rgb = cv2.cvtColor(result_rgb, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL and encode
    pil_image = Image.fromarray(result_rgb)
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return img_str


def generate_comparison_grid(model: nn.Module,
                             images: List[Tuple[torch.Tensor, np.ndarray, int]],
                             labels: List[str],
                             figsize: Tuple[int, int] = (15, 5)) -> np.ndarray:
    """
    Generate a grid comparing original images with their Grad-CAM heatmaps.
    
    Args:
        model: PyTorch model
        images: List of (tensor, original_image, label) tuples
        labels: Class labels ['real', 'fake']
        figsize: Figure size
    
    Returns:
        Grid image as numpy array
    """
    import matplotlib.pyplot as plt
    
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=figsize)
    
    if n == 1:
        axes = axes.reshape(2, 1)
    
    gradcam = GradCAM(model)
    
    for i, (tensor, original, label) in enumerate(images):
        # Original image
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        axes[0, i].imshow(original_rgb)
        axes[0, i].set_title(f"{labels[label]}")
        axes[0, i].axis('off')
        
        # Grad-CAM overlay
        overlay = gradcam.generate_heatmap_overlay(tensor, original)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        axes[1, i].imshow(overlay_rgb)
        axes[1, i].set_title("Grad-CAM")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # Convert figure to image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return img


if __name__ == "__main__":
    # Test Grad-CAM
    print("Testing Grad-CAM implementation")
    
    from module3_model import DeepfakeDetector
    
    # Create model
    model = DeepfakeDetector(backbone='resnet18', pretrained=True)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    dummy_original = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Test Grad-CAM
    gradcam = GradCAM(model)
    heatmap = gradcam(dummy_input)
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    
    # Test overlay
    overlay = gradcam.generate_heatmap_overlay(dummy_input, dummy_original)
    print(f"Overlay shape: {overlay.shape}")
    
    # Test base64 encoding
    b64 = heatmap_to_base64(heatmap, dummy_original)
    print(f"Base64 length: {len(b64)} characters")
    
    print("\nGrad-CAM module loaded successfully!")
