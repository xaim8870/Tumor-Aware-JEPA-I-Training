# src/models/jepa_unet_segmenter.py

from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.jepa_vit import ViTEncoder


def safe_torch_load(path: str, map_location="cpu"):
    """
    Compatibility helper for different PyTorch versions.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        ]

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SimpleUNetDecoder(nn.Module):
    """
    U-Net-style decoder for ViT/JEPA token feature maps.

    Input:
        [B, embed_dim, 16, 16] for 256 image with patch_size 16

    Output:
        [B, 1, 256, 256]
    """

    def __init__(
        self,
        in_channels: int,
        decoder_channels: List[int],
        out_channels: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        blocks = []

        current_channels = in_channels

        for ch in decoder_channels:
            blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        current_channels,
                        ch,
                        kernel_size=2,
                        stride=2,
                    ),
                    ConvBlock(ch, ch, dropout=dropout),
                )
            )

            current_channels = ch

        self.up_blocks = nn.ModuleList(blocks)

        self.head = nn.Conv2d(
            current_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        for block in self.up_blocks:
            x = block(x)

        logits = self.head(x)

        return logits


class JEPAUNetSegmenter(nn.Module):
    """
    Full segmentation model:

        MRI image
            ↓
        JEPA pretrained encoder
            ↓
        token feature representation
            ↓
        reshape to spatial feature map
            ↓
        U-Net-style decoder
            ↓
        tumor mask logits
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 192,
        encoder_depth: int = 4,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        decoder_channels: List[int] | None = None,
    ):
        super().__init__()

        if decoder_channels is None:
            decoder_channels = [128, 64, 32, 16]

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.embed_dim = embed_dim

        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.decoder = SimpleUNetDecoder(
            in_channels=embed_dim,
            decoder_channels=decoder_channels,
            out_channels=1,
            dropout=dropout,
        )

    def load_jepa_encoder(self, checkpoint_path: str):
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"JEPA encoder checkpoint not found: {checkpoint_path}")

        checkpoint = safe_torch_load(str(checkpoint_path), map_location="cpu")

        if isinstance(checkpoint, dict) and "context_encoder_state_dict" in checkpoint:
            encoder_state = checkpoint["context_encoder_state_dict"]

        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            full_state = checkpoint["model_state_dict"]

            encoder_state = {}

            for k, v in full_state.items():
                if k.startswith("context_encoder."):
                    new_key = k.replace("context_encoder.", "", 1)
                    encoder_state[new_key] = v

        else:
            # This handles final_jepa_encoder_100ep.pth,
            # which is usually already only the encoder state dict.
            encoder_state = checkpoint

        missing, unexpected = self.encoder.load_state_dict(
            encoder_state,
            strict=False,
        )

        print(f"Loaded JEPA encoder from: {checkpoint_path}")

        if len(missing) > 0:
            print("Missing encoder keys:")
            for k in missing:
                print("  ", k)

        if len(unexpected) > 0:
            print("Unexpected encoder keys:")
            for k in unexpected:
                print("  ", k)

    def forward(self, images: torch.Tensor):
        tokens = self.encoder(images)
        # [B, N, D]

        B, N, D = tokens.shape

        expected_n = self.grid_size * self.grid_size

        if N != expected_n:
            raise RuntimeError(
                f"Unexpected token count. Got {N}, expected {expected_n}."
            )

        features = tokens.transpose(1, 2).contiguous()
        features = features.view(B, D, self.grid_size, self.grid_size)
        # [B, D, H_patch, W_patch]

        logits = self.decoder(features)

        if logits.shape[-2:] != images.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return logits