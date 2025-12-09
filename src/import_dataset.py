"""
Import Kaggle Dataset for Deepfake Detection

Downloads the 'manjilkarki/deepfake-and-real-images' dataset from Kaggle
and organizes it into the project's data folder structure.

Requirements:
    pip install kagglehub

Usage:
    python src/import_dataset.py
"""

import os
import sys
import shutil
from pathlib import Path

try:
    import kagglehub
except ImportError:
    print("ERROR: kagglehub is not installed.")
    print("Please install it using: pip install kagglehub")
    sys.exit(1)


def download_and_organize_dataset(target_dir='data'):
    """
    Download dataset from Kaggle and organize into train/test structure.

    The dataset structure is:
    downloaded_path/
    ├── Train/
    │   ├── Fake/
    │   └── Real/
    ├── Test/
    │   ├── Fake/
    │   └── Real/
    └── Validation/
        ├── Fake/
        └── Real/

    Args:
        target_dir: Root directory for dataset (default: 'data')
    """
    print("=" * 60)
    print("Kaggle Deepfake Dataset Downloader")
    print("=" * 60)

    # Get project root directory
    project_root = Path(__file__).parent.parent
    target_path = project_root / target_dir

    print(f"\n📂 Target directory: {target_path}")

    # Download dataset from Kaggle
    print("\n📥 Downloading dataset from Kaggle...")
    print("   Dataset: manjilkarki/deepfake-and-real-images")
    print("   (This may take a few minutes depending on your internet speed)")

    try:
        downloaded_path = kagglehub.dataset_download(
            "manjilkarki/deepfake-and-real-images")
        print(f"\n✅ Dataset downloaded to: {downloaded_path}")
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nMake sure you have:")
        print("  1. A Kaggle account")
        print("  2. Kaggle API credentials configured")
        print("  3. Run: kaggle datasets download -d manjilkarki/deepfake-and-real-images")
        sys.exit(1)

    # Organize the dataset
    print("\n📁 Organizing dataset into project structure...")

    downloaded_path = Path(downloaded_path)

    # Create target directories
    train_real = target_path / 'train' / 'real'
    train_fake = target_path / 'train' / 'fake'
    test_real = target_path / 'test' / 'real'
    test_fake = target_path / 'test' / 'fake'

    for dir_path in [train_real, train_fake, test_real, test_fake]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Expected structure: Train/Fake, Train/Real, Test/Fake, Test/Real, Validation/Fake, Validation/Real
    source_train_fake = downloaded_path / 'Train' / 'Fake'
    source_train_real = downloaded_path / 'Train' / 'Real'
    source_test_fake = downloaded_path / 'Test' / 'Fake'
    source_test_real = downloaded_path / 'Test' / 'Real'
    source_val_fake = downloaded_path / 'Validation' / 'Fake'
    source_val_real = downloaded_path / 'Validation' / 'Real'

    # Verify structure exists
    print("   Verifying dataset structure...")
    required_dirs = [
        source_train_fake, source_train_real,
        source_test_fake, source_test_real,
        source_val_fake, source_val_real
    ]

    missing_dirs = [d for d in required_dirs if not d.exists()]
    if missing_dirs:
        print(f"\n⚠️  Warning: Expected dataset structure not found!")
        print(f"   Looking for: Train/Fake, Train/Real, Test/Fake, Test/Real, Validation/Fake, Validation/Real")
        print(f"\n   Missing directories:")
        for d in missing_dirs:
            print(f"   - {d.relative_to(downloaded_path)}")
        print(f"\n   Dataset structure in {downloaded_path}:")
        for item in downloaded_path.rglob('*'):
            if item.is_dir():
                print(f"   - {item.relative_to(downloaded_path)}/")
        sys.exit(1)

    # Count images
    def count_images(directory):
        """Count image files in a directory."""
        if not directory.exists():
            return 0
        return len([f for f in directory.iterdir()
                   if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']])

    train_real_count = count_images(source_train_real)
    train_fake_count = count_images(source_train_fake)
    test_real_count = count_images(source_test_real)
    test_fake_count = count_images(source_test_fake)
    val_real_count = count_images(source_val_real)
    val_fake_count = count_images(source_val_fake)

    print(f"\n📊 Dataset statistics:")
    print(f"   Training:   {train_real_count} real, {train_fake_count} fake")
    print(f"   Testing:    {test_real_count} real, {test_fake_count} fake")
    print(f"   Validation: {val_real_count} real, {val_fake_count} fake")

    # Copy training images
    print("\n📋 Copying training images...")
    copied = 0
    for img_path in source_train_real.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            dest = train_real / img_path.name
            shutil.copy2(img_path, dest)
            copied += 1
            if copied % 100 == 0:
                print(f"   Copied {copied} training real images...")
    print(f"   ✅ Copied {train_real_count} training real images")

    copied = 0
    for img_path in source_train_fake.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            dest = train_fake / img_path.name
            shutil.copy2(img_path, dest)
            copied += 1
            if copied % 100 == 0:
                print(f"   Copied {copied} training fake images...")
    print(f"   ✅ Copied {train_fake_count} training fake images")

    # Copy test images
    print("\n📋 Copying test images...")
    copied = 0
    for img_path in source_test_real.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            dest = test_real / img_path.name
            shutil.copy2(img_path, dest)
            copied += 1
            if copied % 50 == 0:
                print(f"   Copied {copied} test real images...")
    print(f"   ✅ Copied {test_real_count} test real images")

    copied = 0
    for img_path in source_test_fake.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            dest = test_fake / img_path.name
            shutil.copy2(img_path, dest)
            copied += 1
            if copied % 50 == 0:
                print(f"   Copied {copied} test fake images...")
    print(f"   ✅ Copied {test_fake_count} test fake images")

    # Merge validation into test
    print("\n📋 Merging validation images into test set...")
    copied = 0
    for img_path in source_val_real.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            # Add prefix to avoid name conflicts
            dest = test_real / f"val_{img_path.name}"
            shutil.copy2(img_path, dest)
            copied += 1
    print(f"   ✅ Merged {val_real_count} validation real images into test")

    copied = 0
    for img_path in source_val_fake.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            # Add prefix to avoid name conflicts
            dest = test_fake / f"val_{img_path.name}"
            shutil.copy2(img_path, dest)
            copied += 1
    print(f"   ✅ Merged {val_fake_count} validation fake images into test")

    # Final counts
    final_train_real = len(list(train_real.glob('*')))
    final_train_fake = len(list(train_fake.glob('*')))
    final_test_real = len(list(test_real.glob('*')))
    final_test_fake = len(list(test_fake.glob('*')))

    # Summary
    print("\n" + "=" * 60)
    print("✅ Dataset successfully organized!")
    print("=" * 60)
    print(f"\n📁 Final dataset structure:")
    print(f"   {target_path}/")
    print(f"   ├── train/")
    print(f"   │   ├── real/  ({final_train_real} images)")
    print(f"   │   └── fake/  ({final_train_fake} images)")
    print(f"   └── test/")
    print(f"       ├── real/  ({final_test_real} images)")
    print(f"       └── fake/  ({final_test_fake} images)")

    total_images = final_train_real + final_train_fake + \
        final_test_real + final_test_fake
    print(f"\n🎯 Total images: {total_images}")
    print(
        f"   Training: {final_train_real + final_train_fake} ({((final_train_real + final_train_fake)/total_images*100):.1f}%)")
    print(
        f"   Testing:  {final_test_real + final_test_fake} ({((final_test_real + final_test_fake)/total_images*100):.1f}%)")

    print(f"\n📍 Original downloaded location: {downloaded_path}")
    print(f"📍 Organized dataset location: {target_path}")

    print("\n▶️  Next steps:")
    print("   1. Train classical model: python src/train_classical.py")
    print("   2. Train deep learning model: python src/train.py --epochs 20")
    print("   3. Run web app: python app/app.py")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Download and organize Kaggle deepfake dataset'
    )
    parser.add_argument(
        '--target_dir',
        type=str,
        default='data',
        help='Target directory for dataset (default: data)'
    )

    args = parser.parse_args()

    try:
        download_and_organize_dataset(args.target_dir)
    except KeyboardInterrupt:
        print("\n\n❌ Download cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
