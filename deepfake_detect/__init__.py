"""Face deepfake detection for images and videos: frozen CLIP features + a linear probe."""
from .model import Detector

__all__ = ["Detector"]
__version__ = "0.1.0"
