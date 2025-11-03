import os
from pathlib import Path
from typing import List, Tuple
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

"""
MVTec helper:
- Expects a directory structure like: data_root/category/{train,test,ground_truth}/...
- We only need image-level for this starter: train/good for feature bank, test/* for scoring.
- Label convention: 0 = good (normal), 1 = defective (anomaly).
"""

def list_mvtec_images(data_root: str, category: str):
    data_root = Path(data_root)
    cat_root = data_root / category
    assert cat_root.exists(), f"Category folder not found: {cat_root}"

    # Train: only 'good'
    train_dir = cat_root / "train" / "good"
    train_paths = sorted([p for p in train_dir.rglob("*.png")] + [p for p in train_dir.rglob("*.jpg")])
    train_labels = [0] * len(train_paths)  # all normal

    # Test: mix of 'good' and defect types
    test_root = cat_root / "test"
    test_paths, test_labels = [], []
    for sub in sorted(test_root.iterdir()):
        if not sub.is_dir():
            continue
        imgs = sorted([p for p in sub.rglob("*.png")] + [p for p in sub.rglob("*.jpg")])
        if sub.name == "good":
            test_paths += imgs
            test_labels += [0] * len(imgs)
        else:
            test_paths += imgs
            test_labels += [1] * len(imgs)

    return train_paths, train_labels, test_paths, test_labels

class MVTECImages(Dataset):
    def __init__(self, paths: List[Path], labels: List[int], image_size: int = 256):
        self.paths = paths
        self.labels = labels
        # Simple transform: resize + center-crop + normalize (ImageNet stats for ResNet18 backbone)
        self.tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        y = self.labels[idx]
        img = Image.open(p).convert("RGB")
        x = self.tf(img)
        return x, y, str(p)
