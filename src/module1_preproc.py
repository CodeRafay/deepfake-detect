"""
Module 1: Image Preprocessing and Restoration

This module provides functions for:
- Image degradation (noise, compression)
- Geometric transforms (rotation, scaling)
- Intensity transforms (gamma, histogram equalization)
- Image restoration (denoising)
- Quality metrics (PSNR, SSIM)
"""

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from typing import Tuple, Optional


# ============================================================================
# DEGRADATION FUNCTIONS
# ============================================================================

def add_gaussian_noise(img: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """
    Add Gaussian noise to an image.

    Args:
        img: Input image (uint8, BGR or grayscale)
        sigma: Standard deviation of noise (0-255 scale)

    Returns:
        Noisy image (uint8)
    """
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(img: np.ndarray, amount: float = 0.02) -> np.ndarray:
    """
    Add salt and pepper noise to an image.

    Args:
        img: Input image (uint8)
        amount: Proportion of pixels to affect (0-1)

    Returns:
        Noisy image (uint8)
    """
    noisy = img.copy()

    # Salt (white pixels)
    num_salt = int(amount * img.size / 2)
    coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]
    if len(img.shape) == 3:
        noisy[coords[0], coords[1], :] = 255
    else:
        noisy[coords[0], coords[1]] = 255

    # Pepper (black pixels)
    num_pepper = int(amount * img.size / 2)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape[:2]]
    if len(img.shape) == 3:
        noisy[coords[0], coords[1], :] = 0
    else:
        noisy[coords[0], coords[1]] = 0

    return noisy


def jpeg_compress(img: np.ndarray, quality: int = 30) -> np.ndarray:
    """
    Simulate JPEG compression artifacts.

    Args:
        img: Input image (uint8, BGR)
        quality: JPEG quality (1-100, lower = more compression)

    Returns:
        Compressed image (uint8)
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]  # Set JPEG quality
    # Encode to JPEG format
    _, encoded = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)  # Decode back to BGR image


def add_blur(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Add Gaussian blur to an image.

    Args:
        img: Input image
        kernel_size: Size of blur kernel (odd number)

    Returns:
        Blurred image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


# ============================================================================
# GEOMETRIC TRANSFORMS
# ============================================================================

def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate image by given angle.

    Args:
        img: Input image
        angle: Rotation angle in degrees (positive = counter-clockwise)

    Returns:
        Rotated image (same size, with black borders)
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def scale_image(img: np.ndarray, scale: float) -> np.ndarray:
    """
    Scale image by given factor.

    Args:
        img: Input image
        scale: Scale factor (e.g., 0.5 = half size, 2.0 = double size)

    Returns:
        Scaled image
    """
    h, w = img.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)


def flip_image(img: np.ndarray, direction: str = 'horizontal') -> np.ndarray:
    """
    Flip image horizontally or vertically.

    Args:
        img: Input image
        direction: 'horizontal' or 'vertical'

    Returns:
        Flipped image
    """
    if direction == 'horizontal':
        return cv2.flip(img, 1)
    elif direction == 'vertical':
        return cv2.flip(img, 0)
    else:
        raise ValueError("direction must be 'horizontal' or 'vertical'")


def crop_center(img: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
    """
    Crop the center of an image.

    Args:
        img: Input image
        crop_size: (width, height) of crop

    Returns:
        Cropped image
    """
    h, w = img.shape[:2]
    crop_w, crop_h = crop_size

    start_x = max(0, (w - crop_w) // 2)
    start_y = max(0, (h - crop_h) // 2)

    return img[start_y:start_y + crop_h, start_x:start_x + crop_w]


# ============================================================================
# INTENSITY TRANSFORMS
# ============================================================================

def gamma_correction(img: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """
    Apply gamma correction for brightness adjustment.

    Args:
        img: Input image (uint8)
        gamma: Gamma value (>1 = darker, <1 = brighter)

    Returns:
        Gamma-corrected image
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def histogram_equalization(img: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization for contrast enhancement.

    Args:
        img: Input image (BGR or grayscale)

    Returns:
        Contrast-enhanced image
    """
    if len(img.shape) == 3:
        # Convert to YUV, equalize Y channel
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        return cv2.equalizeHist(img)


def clahe_enhancement(img: np.ndarray, clip_limit: float = 2.0,
                      tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        img: Input image
        clip_limit: Threshold for contrast limiting
        tile_size: Size of grid for histogram equalization

    Returns:
        Enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        return clahe.apply(img)


def adjust_brightness_contrast(img: np.ndarray, brightness: float = 0,
                               contrast: float = 1.0) -> np.ndarray:
    """
    Adjust brightness and contrast.

    Args:
        img: Input image
        brightness: Brightness adjustment (-127 to 127)
        contrast: Contrast multiplier (0.5 to 2.0)

    Returns:
        Adjusted image
    """
    adjusted = img.astype(np.float32)
    adjusted = adjusted * contrast + brightness
    return np.clip(adjusted, 0, 255).astype(np.uint8)


# ============================================================================
# RESTORATION FUNCTIONS
# ============================================================================

def denoise_nlm(img: np.ndarray, h: float = 10,
                template_size: int = 7, search_size: int = 21) -> np.ndarray:
    """
    Denoise using Non-Local Means algorithm.

    Args:
        img: Input noisy image
        h: Filter strength (higher = more denoising but more blur)
        template_size: Template patch size (should be odd)
        search_size: Search window size (should be odd)

    Returns:
        Denoised image
    """
    if len(img.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(img, None, h, h,
                                               template_size, search_size)
    else:
        return cv2.fastNlMeansDenoising(img, None, h,
                                        template_size, search_size)


def denoise_bilateral(img: np.ndarray, d: int = 9,
                      sigma_color: float = 75,
                      sigma_space: float = 75) -> np.ndarray:
    """
    Denoise using bilateral filtering (edge-preserving).

    Args:
        img: Input image
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space

    Returns:
        Denoised image
    """
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def denoise_median(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Denoise using median filter (good for salt-pepper noise).

    Args:
        img: Input image
        kernel_size: Size of median filter kernel (odd number)

    Returns:
        Denoised image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(img, kernel_size)


def sharpen_image(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Sharpen image using unsharp masking.

    Args:
        img: Input image
        strength: Sharpening strength

    Returns:
        Sharpened image
    """
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    sharpened = cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)
    return sharpened


# ============================================================================
# QUALITY METRICS
# ============================================================================

def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between two images.

    Args:
        original: Original image
        processed: Processed/degraded image

    Returns:
        PSNR value in dB (higher is better)
    """
    # Ensure same size and type
    if original.shape != processed.shape:
        processed = cv2.resize(
            processed, (original.shape[1], original.shape[0]))

    return peak_signal_noise_ratio(original, processed)


def compute_ssim(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Compute Structural Similarity Index between two images.

    Args:
        original: Original image
        processed: Processed/degraded image

    Returns:
        SSIM value (0-1, higher is better)
    """
    # Ensure same size
    if original.shape != processed.shape:
        processed = cv2.resize(
            processed, (original.shape[1], original.shape[0]))

    # Convert to grayscale if color
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        processed_gray = processed

    return structural_similarity(original_gray, processed_gray)


def compute_mse(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Compute Mean Squared Error between two images.

    Args:
        original: Original image
        processed: Processed/degraded image

    Returns:
        MSE value (lower is better)
    """
    if original.shape != processed.shape:
        processed = cv2.resize(
            processed, (original.shape[1], original.shape[0]))

    return np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)


# ============================================================================
# PREPROCESSING PIPELINE
# ============================================================================

def preprocess_for_classification(img: np.ndarray,
                                  target_size: Tuple[int, int] = (224, 224),
                                  normalize: bool = True) -> np.ndarray:
    """
    Preprocess image for classification model.

    Args:
        img: Input image (BGR)
        target_size: Target (width, height)
        normalize: Whether to normalize to [0, 1]

    Returns:
        Preprocessed image
    """
    # Resize
    processed = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

    # Convert BGR to RGB
    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    if normalize:
        processed = processed.astype(np.float32) / 255.0

    return processed


def apply_preprocessing_pipeline(img: np.ndarray,
                                 pipeline: list) -> np.ndarray:
    """
    Apply a sequence of preprocessing operations.

    Args:
        img: Input image
        pipeline: List of (function, kwargs) tuples

    Returns:
        Processed image

    Example:
        pipeline = [
            (add_gaussian_noise, {'sigma': 15}),
            (denoise_nlm, {'h': 10}),
        ]
        result = apply_preprocessing_pipeline(img, pipeline)
    """
    result = img.copy()
    for func, kwargs in pipeline:
        result = func(result, **kwargs)
    return result


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def create_comparison_grid(images: list, titles: list,
                           cols: int = 3) -> np.ndarray:
    """
    Create a grid of images for visual comparison.

    Args:
        images: List of images
        titles: List of titles (not rendered, for reference)
        cols: Number of columns in grid

    Returns:
        Grid image
    """
    n = len(images)
    rows = (n + cols - 1) // cols

    # Get max dimensions
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)

    # Create grid
    grid = np.zeros((rows * max_h, cols * max_w, 3), dtype=np.uint8)

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols

        # Resize if needed
        if img.shape[0] != max_h or img.shape[1] != max_w:
            img = cv2.resize(img, (max_w, max_h))

        # Handle grayscale
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        y_start = row * max_h
        x_start = col * max_w
        grid[y_start:y_start + max_h, x_start:x_start + max_w] = img

    return grid


if __name__ == "__main__":
    # Quick demo
    import os

    # Create a test image
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Test degradations
    noisy = add_gaussian_noise(test_img, sigma=25)
    compressed = jpeg_compress(test_img, quality=20)

    # Test restoration
    denoised = denoise_nlm(noisy, h=10)

    # Compute metrics
    print(f"Noisy PSNR: {compute_psnr(test_img, noisy):.2f} dB")
    print(f"Noisy SSIM: {compute_ssim(test_img, noisy):.4f}")
    print(f"Denoised PSNR: {compute_psnr(test_img, denoised):.2f} dB")
    print(f"Denoised SSIM: {compute_ssim(test_img, denoised):.4f}")
    print(f"Compressed PSNR: {compute_psnr(test_img, compressed):.2f} dB")
    print(f"Compressed SSIM: {compute_ssim(test_img, compressed):.4f}")

    print("\nModule 1 preprocessing functions loaded successfully!")
