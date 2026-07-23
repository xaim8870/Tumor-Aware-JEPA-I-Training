# scripts/check_dataset.py

import sys
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

# Allow imports from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.datasets.brisc_jepa_dataset import BRISCJEPADataset


def main():
    config_path = ROOT / "configs" / "pretrain_jepa.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg["data"]["data_root"])
    train_dir = data_root / cfg["data"]["train_images"]
    val_dir = data_root / cfg["data"]["val_images"]

    img_size = cfg["data"]["img_size"]
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]

    print("=" * 60)
    print("Checking JEPA dataset")
    print("=" * 60)

    print(f"Data root: {data_root}")
    print(f"Train image dir: {train_dir}")
    print(f"Val image dir:   {val_dir}")
    print(f"Image size: {img_size}")

    train_dataset = BRISCJEPADataset(train_dir, img_size=img_size)
    val_dataset = BRISCJEPADataset(val_dir, img_size=img_size)

    print(f"\nTrain images: {len(train_dataset)}")
    print(f"Val images:   {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    images, names = next(iter(train_loader))

    print("\nOne batch loaded successfully.")
    print(f"Batch image tensor shape: {images.shape}")
    print(f"Example filenames: {names[:3]}")

    print("\nExpected shape:")
    print(f"[batch_size, 1, {img_size}, {img_size}]")

    print("\nDataset check complete.")


if __name__ == "__main__":
    main()