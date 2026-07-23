# src/datasets/brisc_seg_dataset.py

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class BRISCSegmentationDataset(Dataset):
    """
    BRISC binary segmentation dataset.

    Output:
        image: [1, H, W], normalized to [-1, 1]
        mask:  [1, H, W], binary float 0/1
        name:  image filename
    """

    def __init__(
        self,
        image_dir,
        mask_dir,
        img_size=256,
        mask_threshold=0,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.mask_threshold = mask_threshold

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        self.image_paths = sorted(
            [
                p for p in self.image_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            ]
        )

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in: {self.image_dir}")

        self.samples = []

        for img_path in self.image_paths:
            mask_path = self._find_mask_for_image(img_path)

            if mask_path is None:
                raise FileNotFoundError(
                    f"No matching mask found for image:\n"
                    f"Image: {img_path}\n"
                    f"Mask dir: {self.mask_dir}"
                )

            self.samples.append((img_path, mask_path))

        self.image_transform = T.Compose(
            [
                T.Grayscale(num_output_channels=1),
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def _find_mask_for_image(self, img_path: Path):
        stem = img_path.stem
        suffix = img_path.suffix

        candidates = [
            self.mask_dir / img_path.name,
            self.mask_dir / f"{stem}.png",
            self.mask_dir / f"{stem}.jpg",
            self.mask_dir / f"{stem}.jpeg",
            self.mask_dir / f"{stem}.bmp",
            self.mask_dir / f"{stem}.tif",
            self.mask_dir / f"{stem}.tiff",
            self.mask_dir / f"{stem}_mask.png",
            self.mask_dir / f"{stem}_mask{suffix}",
            self.mask_dir / f"{stem}_seg.png",
            self.mask_dir / f"{stem}_seg{suffix}",
            self.mask_dir / f"{stem}_label.png",
            self.mask_dir / f"{stem}_label{suffix}",
        ]

        for c in candidates:
            if c.exists():
                return c

        return None

    def __len__(self):
        return len(self.samples)

    def _load_mask(self, mask_path: Path):
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)

        mask_np = np.array(mask)

        # Robust binary conversion.
        # This works for both 0/1 masks and 0/255 masks.
        mask_bin = (mask_np > 0).astype(np.float32)

        mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0)

        return mask_tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("L")
        image = self.image_transform(image)

        mask = self._load_mask(mask_path)

        return image, mask, img_path.name