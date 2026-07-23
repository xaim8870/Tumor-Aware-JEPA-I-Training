# scripts/check_segmentation_batch.py

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Change this import if your segmentation dataset class has another name
from src.datasets.brisc_seg_dataset import BRISCSegmentationDataset


def main():
    data_root = Path(
        "D:/Brain Tumor Segmentation/data/processed/brisc_segformer_binary"
    )

    train_images = data_root / "images" / "train"
    train_masks = data_root / "masks" / "train"

    dataset = BRISCSegmentationDataset(
        image_dir=train_images,
        mask_dir=train_masks,
        img_size=256,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    images, masks, names = next(iter(loader))

    print("=" * 60)
    print("Segmentation Batch Check")
    print("=" * 60)

    print("Image shape:", images.shape)
    print("Mask shape: ", masks.shape)

    print("\nImage min/max:")
    print(images.min().item(), images.max().item())

    print("\nMask min/max:")
    print(masks.min().item(), masks.max().item())

    print("\nUnique mask values:")
    print(torch.unique(masks))

    fg_pixels = (masks > 0.5).sum().item()
    total_pixels = masks.numel()
    fg_ratio = fg_pixels / total_pixels

    print("\nForeground tumor pixels:", fg_pixels)
    print("Total pixels:", total_pixels)
    print("Foreground ratio:", fg_ratio)

    print("\nExample names:")
    for n in names:
        print(n)


if __name__ == "__main__":
    main()