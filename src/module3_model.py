"""
Module 3: Deep Learning Model for Deepfake Detection

This module provides:
- DeepfakeDetector: ResNet18/EfficientNet-based binary classifier
- Data augmentation and loading utilities
- Transfer learning with freezing options
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import numpy as np
import os
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path


class DeepfakeDetector(nn.Module):
    """
    Deep learning model for deepfake detection.

    Uses transfer learning from ImageNet-pretrained models with a
    custom classification head for binary detection.
    """

    def __init__(self,
                 backbone: str = 'resnet18',
                 pretrained: bool = True,
                 dropout: float = 0.5,
                 freeze_backbone: bool = False):
        """
        Initialize the detector.

        Args:
            backbone: Model backbone ('resnet18', 'resnet34', 'efficientnet_b0')
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate in classification head
            freeze_backbone: Whether to freeze backbone weights
        """
        super(DeepfakeDetector, self).__init__()

        self.backbone_name = backbone

        if backbone == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif backbone == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif backbone == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 1)
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze backbone weights for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone weights."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Logits tensor (B, 1)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        return self.backbone(x)


class DeepfakeDataset(Dataset):
    """
    Dataset for deepfake detection.

    Expects directory structure:
        root/
            real/
                img1.jpg
                ...
            fake/
                img1.jpg
                ...
    """

    def __init__(self,
                 root_dir: str,
                 transform: Optional[transforms.Compose] = None,
                 max_samples_per_class: Optional[int] = None):
        """
        Initialize dataset.

        Args:
            root_dir: Root directory with real/ and fake/ subdirs
            transform: Image transforms to apply
            max_samples_per_class: Limit samples per class (for quick testing)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.samples = []
        self.labels = []

        # Check for both capitalized and lowercase folder names
        real_dir = self.root_dir / \
            'Real' if (self.root_dir /
                       'Real').exists() else self.root_dir / 'real'
        fake_dir = self.root_dir / \
            'Fake' if (self.root_dir /
                       'Fake').exists() else self.root_dir / 'fake'

        # Load real images (label = 0)
        if real_dir.exists():
            real_images = self._get_image_files(real_dir)
            if max_samples_per_class:
                real_images = real_images[:max_samples_per_class]
            self.samples.extend(real_images)
            self.labels.extend([0] * len(real_images))

        # Load fake images (label = 1)
        if fake_dir.exists():
            fake_images = self._get_image_files(fake_dir)
            if max_samples_per_class:
                fake_images = fake_images[:max_samples_per_class]
            self.samples.extend(fake_images)
            self.labels.extend([1] * len(fake_images))

        # Convert to numpy for indexing
        self.labels = np.array(self.labels)

    def _get_image_files(self, directory: Path) -> List[str]:
        """Get all image files in directory."""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        files = set()  # Use set to avoid duplicates on case-insensitive filesystems
        for ext in extensions:
            files.update(directory.glob(f'*{ext}'))
            files.update(directory.glob(f'*{ext.upper()}'))
        return sorted([str(f) for f in files])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.samples[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """Get class weights for balanced sampling."""
        _, counts = np.unique(self.labels, return_counts=True)
        weights = 1.0 / counts
        sample_weights = weights[self.labels]
        return torch.FloatTensor(sample_weights)


def get_transforms(image_size: int = 224,
                   is_training: bool = True,
                   augment_config: Optional[Dict] = None) -> transforms.Compose:
    """
    Get image transforms for training or inference.

    Args:
        image_size: Target image size
        is_training: If True, apply data augmentation
        augment_config: Augmentation configuration dict

    Returns:
        Compose transform
    """
    # ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if is_training:
        # Default augmentation config
        if augment_config is None:
            augment_config = {
                'horizontal_flip': True,
                'rotation': 15,
                'color_jitter': {
                    'brightness': 0.2,
                    'contrast': 0.2,
                    'saturation': 0.2,
                    'hue': 0.1
                }
            }

        transform_list = [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
        ]

        if augment_config.get('horizontal_flip', True):
            transform_list.append(transforms.RandomHorizontalFlip())

        if augment_config.get('rotation', 0) > 0:
            transform_list.append(
                transforms.RandomRotation(augment_config['rotation'])
            )

        if augment_config.get('color_jitter'):
            cj = augment_config['color_jitter']
            transform_list.append(
                transforms.ColorJitter(
                    brightness=cj.get('brightness', 0),
                    contrast=cj.get('contrast', 0),
                    saturation=cj.get('saturation', 0),
                    hue=cj.get('hue', 0)
                )
            )

        transform_list.extend([
            transforms.ToTensor(),
            normalize
        ])

        return transforms.Compose(transform_list)

    else:
        # Validation/inference transforms
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize
        ])


def create_data_loaders(train_dir: str,
                        test_dir: str,
                        batch_size: int = 32,
                        image_size: int = 224,
                        num_workers: int = 4,
                        balanced_sampling: bool = True,
                        max_samples_per_class: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and test data loaders.

    Args:
        train_dir: Training data directory
        test_dir: Test data directory
        batch_size: Batch size
        image_size: Image size
        num_workers: Number of data loading workers
        balanced_sampling: Use weighted random sampler
        max_samples_per_class: Limit samples per class

    Returns:
        Training and test DataLoaders
    """
    # Create datasets
    train_transform = get_transforms(image_size, is_training=True)
    test_transform = get_transforms(image_size, is_training=False)

    train_dataset = DeepfakeDataset(
        train_dir, train_transform, max_samples_per_class)
    test_dataset = DeepfakeDataset(
        test_dir, test_transform, max_samples_per_class)

    # Create sampler for balanced sampling
    if balanced_sampling and len(train_dataset) > 0:
        sample_weights = train_dataset.get_class_weights()
        sampler = WeightedRandomSampler(sample_weights, len(train_dataset))
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  sampler=sampler, num_workers=num_workers,
                                  pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  pin_memory=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    return train_loader, test_loader


def save_model(model: nn.Module,
               path: str,
               optimizer: Optional[torch.optim.Optimizer] = None,
               epoch: Optional[int] = None,
               metrics: Optional[Dict] = None) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        path: Save path
        optimizer: Optimizer state (optional)
        epoch: Current epoch (optional)
        metrics: Training metrics (optional)
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(
        path) else '.', exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'backbone': model.backbone_name,
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    if metrics is not None:
        checkpoint['metrics'] = metrics

    torch.save(checkpoint, path)


def load_model(path: str,
               device: Optional[torch.device] = None) -> DeepfakeDetector:
    """
    Load model from checkpoint.

    Args:
        path: Checkpoint path
        device: Device to load to

    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(path, map_location=device)

    backbone = checkpoint.get('backbone', 'resnet18')
    model = DeepfakeDetector(backbone=backbone, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


if __name__ == "__main__":
    # Test model
    print("Testing Module 3: Deep Learning Model")

    # Create model
    model = DeepfakeDetector(backbone='resnet18', pretrained=True)
    print(f"Model created: {model.backbone_name}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Test transforms
    train_transform = get_transforms(224, is_training=True)
    test_transform = get_transforms(224, is_training=False)
    print(f"\nTrain transforms: {train_transform}")
    print(f"\nTest transforms: {test_transform}")

    print("\nModule 3 loaded successfully!")
