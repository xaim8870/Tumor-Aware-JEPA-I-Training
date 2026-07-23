# src/datasets/brisc_jepa_dataset.py

from pathlib import Path
from typing import Tuple

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class BRISCJEPADataset(Dataset):
    """
    Dataset for JEPA pretraining.

    This loader uses MRI images only.
    It does NOT use masks because JEPA pretraining is self-supervised.
    """

    def __init__(self, image_dir: str, img_size: int = 256):
        self.image_dir = Path(image_dir)

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        self.image_paths = sorted(
            list(self.image_dir.glob("*.png"))
            + list(self.image_dir.glob("*.jpg"))
            + list(self.image_dir.glob("*.jpeg"))
            + list(self.image_dir.glob("*.bmp"))
            + list(self.image_dir.glob("*.tif"))
            + list(self.image_dir.glob("*.tiff"))
        )

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in: {self.image_dir}")

        self.transform = T.Compose(
            [
                T.Grayscale(num_output_channels=1),
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple:
        img_path = self.image_paths[idx]

        image = Image.open(img_path).convert("L")
        image = self.transform(image)

        return image, img_path.name