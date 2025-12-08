"""
Utility functions for the deepfake detection pipeline.
"""

import os
import random
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import cv2
import matplotlib.pyplot as plt
from datetime import datetime


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to YAML file."""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def load_image(path: str, color: bool = True) -> Optional[np.ndarray]:
    """
    Load an image from disk.
    
    Args:
        path: Path to image file
        color: If True, load as BGR color image
    
    Returns:
        Image as numpy array or None if failed
    """
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(path, flag)
    if img is None:
        print(f"Warning: Could not load image: {path}")
    return img


def save_image(img: np.ndarray, path: str) -> bool:
    """
    Save an image to disk.
    
    Args:
        img: Image array
        path: Output path
    
    Returns:
        True if successful
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return cv2.imwrite(path, img)


def get_image_files(directory: str, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')) -> List[str]:
    """
    Get all image files in a directory.
    
    Args:
        directory: Directory to search
        extensions: Tuple of valid extensions
    
    Returns:
        List of image file paths
    """
    files = []
    directory = Path(directory)
    
    if not directory.exists():
        return files
    
    for ext in extensions:
        files.extend(directory.glob(f'*{ext}'))
        files.extend(directory.glob(f'*{ext.upper()}'))
    
    return sorted([str(f) for f in files])


def create_directory(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def get_timestamp() -> str:
    """Get current timestamp as string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_figure(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    """Save matplotlib figure to file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, 
                    alpha: float = 0.5) -> np.ndarray:
    """
    Overlay a heatmap on an image.
    
    Args:
        image: Original image (BGR)
        heatmap: Heatmap array (0-1 or 0-255)
        alpha: Blending factor
    
    Returns:
        Overlaid image
    """
    # Normalize heatmap to 0-255
    if heatmap.max() <= 1.0:
        heatmap = (heatmap * 255).astype(np.uint8)
    else:
        heatmap = heatmap.astype(np.uint8)
    
    # Resize heatmap to match image
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay


def calculate_class_weights(labels: List[int]) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        labels: List of class labels
    
    Returns:
        Tensor of class weights
    """
    classes, counts = np.unique(labels, return_counts=True)
    weights = 1.0 / counts
    weights = weights / weights.sum()
    return torch.FloatTensor(weights)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, 
                 mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class MetricsLogger:
    """Simple metrics logger to CSV."""
    
    def __init__(self, log_dir: str, filename: str = 'metrics.csv'):
        self.log_path = os.path.join(log_dir, filename)
        create_directory(log_dir)
        self.headers_written = False
    
    def log(self, metrics: Dict[str, float], epoch: int) -> None:
        """Log metrics for an epoch."""
        metrics['epoch'] = epoch
        
        if not self.headers_written:
            with open(self.log_path, 'w') as f:
                f.write(','.join(metrics.keys()) + '\n')
            self.headers_written = True
        
        with open(self.log_path, 'a') as f:
            f.write(','.join(str(v) for v in metrics.values()) + '\n')


if __name__ == "__main__":
    # Test utilities
    print(f"Device: {get_device()}")
    set_seed(42)
    print("Random seed set successfully")
    print(f"Timestamp: {get_timestamp()}")
    print("\nUtility functions loaded successfully!")
