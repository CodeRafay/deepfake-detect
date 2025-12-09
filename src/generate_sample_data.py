"""
Sample Data Generator for Testing

Generates synthetic "real" and "fake" images for testing the pipeline
when a real dataset is not available.

This creates:
- Real images: Natural face-like patterns
- Fake images: Slightly modified versions with artificial artifacts

Note: This is for testing purposes only. For real deepfake detection,
use a proper dataset like FaceForensics++ or Celeb-DF.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random


def generate_face_like_pattern(size: int = 224) -> np.ndarray:
    """Generate a face-like pattern for testing."""
    img = np.ones((size, size, 3), dtype=np.uint8) * 200

    # Add skin-like color gradient
    for y in range(size):
        for x in range(size):
            # Warm skin tone gradient
            base_color = [180 + random.randint(-10, 10),
                          150 + random.randint(-10, 10),
                          130 + random.randint(-10, 10)]
            img[y, x] = base_color

    # Add oval face shape
    center = (size // 2, size // 2)
    axes = (size // 3, size // 2 - 20)
    cv2.ellipse(img, center, axes, 0, 0, 360, (210, 180, 160), -1)

    # Add eyes (simple circles)
    eye_y = size // 2 - 20
    left_eye = (size // 2 - 30, eye_y)
    right_eye = (size // 2 + 30, eye_y)
    cv2.circle(img, left_eye, 15, (255, 255, 255), -1)
    cv2.circle(img, right_eye, 15, (255, 255, 255), -1)
    cv2.circle(img, left_eye, 7, (50, 50, 50), -1)
    cv2.circle(img, right_eye, 7, (50, 50, 50), -1)

    # Add nose
    nose_points = np.array([
        [size // 2, size // 2],
        [size // 2 - 10, size // 2 + 30],
        [size // 2 + 10, size // 2 + 30]
    ])
    cv2.polylines(img, [nose_points], True, (180, 150, 130), 2)

    # Add mouth
    cv2.ellipse(img, (size // 2, size // 2 + 50),
                (25, 10), 0, 0, 180, (150, 100, 100), 2)

    # Add some texture noise
    noise = np.random.normal(0, 5, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Slight blur for realism
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


def add_fake_artifacts(img: np.ndarray) -> np.ndarray:
    """Add artifacts that simulate deepfake manipulation."""
    fake = img.copy()
    h, w = fake.shape[:2]

    # 1. Add blending boundary artifacts
    mask = np.zeros((h, w), dtype=np.float32)
    center = (w // 2, h // 2)
    cv2.ellipse(mask, center, (w // 3, h // 2 - 30), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)

    # Create visible edge artifacts
    edge_mask = cv2.Canny((mask * 255).astype(np.uint8), 50, 100)
    edge_mask = cv2.dilate(edge_mask, np.ones((3, 3), np.uint8))
    fake[edge_mask > 0] = fake[edge_mask > 0] * 0.8

    # 2. Add color inconsistency in face region
    face_region = mask > 0.5
    if np.sum(face_region) > 0:
        # Shift hue slightly
        hsv = cv2.cvtColor(fake, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = np.where(face_region, hsv[:, :, 0] + 5, hsv[:, :, 0])
        hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)
        fake = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 3. Add compression artifacts in random blocks
    block_size = 16
    for _ in range(5):
        bx = random.randint(0, w - block_size)
        by = random.randint(0, h - block_size)
        block = fake[by:by+block_size, bx:bx+block_size]
        # Simulate DCT block artifacts
        block = cv2.resize(cv2.resize(block, (4, 4)), (block_size, block_size))
        fake[by:by+block_size, bx:bx+block_size] = block

    # 4. Add slight warping artifacts around eyes/mouth
    eye_y = h // 2 - 20
    for eye_x in [w // 2 - 30, w // 2 + 30]:
        roi = fake[eye_y-20:eye_y+20, eye_x-20:eye_x+20].copy()
        if roi.shape[0] > 0 and roi.shape[1] > 0:
            # Add slight distortion
            rows, cols = roi.shape[:2]
            for i in range(rows):
                for j in range(cols):
                    offset_x = int(2 * np.sin(i / 5))
                    offset_y = int(2 * np.cos(j / 5))
                    ni, nj = min(max(i + offset_y, 0), rows -
                                 1), min(max(j + offset_x, 0), cols-1)
                    roi[i, j] = roi[ni, nj]
            fake[eye_y-20:eye_y+20, eye_x-20:eye_x+20] = roi

    # 5. Add unnatural smoothness in parts (face swap often overly smooth)
    smooth_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(smooth_mask, (w//2, h//2 + 20), (40, 30), 0, 0, 360, 255, -1)
    smooth_region = cv2.GaussianBlur(fake, (9, 9), 0)
    fake = np.where(smooth_mask[:, :, np.newaxis] > 128,
                    smooth_region.astype(np.uint8),
                    fake)

    return fake


def generate_sample_dataset(output_dir: str,
                            num_train_each: int = 100,
                            num_test_each: int = 25,
                            image_size: int = 224) -> None:
    """
    Generate a sample dataset for testing.

    Args:
        output_dir: Output directory
        num_train_each: Number of training images per class
        num_test_each: Number of test images per class
        image_size: Image size
    """
    output_path = Path(output_dir)

    # Create directories
    for split in ['train', 'test']:
        for cls in ['real', 'fake']:
            (output_path / split / cls).mkdir(parents=True, exist_ok=True)

    print("Generating sample dataset...")
    print(f"Train: {num_train_each} real + {num_train_each} fake")
    print(f"Test: {num_test_each} real + {num_test_each} fake")

    # Generate training data
    print("\nGenerating training data...")
    for i in tqdm(range(num_train_each), desc="Train real"):
        img = generate_face_like_pattern(image_size)
        cv2.imwrite(str(output_path / 'train' /
                    'real' / f'real_{i:04d}.jpg'), img)

    for i in tqdm(range(num_train_each), desc="Train fake"):
        img = generate_face_like_pattern(image_size)
        fake = add_fake_artifacts(img)
        cv2.imwrite(str(output_path / 'train' /
                    'fake' / f'fake_{i:04d}.jpg'), fake)

    # Generate test data
    print("\nGenerating test data...")
    for i in tqdm(range(num_test_each), desc="Test real"):
        img = generate_face_like_pattern(image_size)
        cv2.imwrite(str(output_path / 'test' /
                    'real' / f'real_{i:04d}.jpg'), img)

    for i in tqdm(range(num_test_each), desc="Test fake"):
        img = generate_face_like_pattern(image_size)
        fake = add_fake_artifacts(img)
        cv2.imwrite(str(output_path / 'test' /
                    'fake' / f'fake_{i:04d}.jpg'), fake)

    print(f"\nDataset generated at: {output_path}")
    print(f"Total images: {2 * (num_train_each + num_test_each)}")


def download_sample_faces(output_dir: str, num_images: int = 50) -> bool:
    """
    Attempt to download sample face images from a public source.
    Falls back to synthetic generation if download fails.

    Returns:
        True if download successful, False if fell back to synthetic
    """
    print("Note: Using synthetic data generator (no external download)")
    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate sample dataset')
    parser.add_argument('--output', type=str, default='data',
                        help='Output directory')
    parser.add_argument('--train', type=int, default=100,
                        help='Number of training images per class')
    parser.add_argument('--test', type=int, default=25,
                        help='Number of test images per class')
    parser.add_argument('--size', type=int, default=224,
                        help='Image size')

    args = parser.parse_args()

    generate_sample_dataset(
        output_dir=args.output,
        num_train_each=args.train,
        num_test_each=args.test,
        image_size=args.size
    )
