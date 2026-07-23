# src/models/jepa_vit.py

import copy
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    Converts image into patch tokens using Conv2d.

    Input:
        [B, C, H, W]

    Output:
        [B, N, D]
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 192,
    ):
        super().__init__()

        assert img_size % patch_size == 0

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # [B, D, H_p, W_p]
        x = x.flatten(2).transpose(1, 2)
        # [B, N, D]
        return x


class ViTEncoder(nn.Module):
    """
    Small ViT-style encoder for JEPA pretraining.
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        self.num_patches = self.patch_embed.num_patches
        self.embed_dim = embed_dim

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.blocks = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        images: torch.Tensor,
        token_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self.patch_embed(images)
        tokens = tokens + self.pos_embed

        if token_indices is not None:
            idx = token_indices.unsqueeze(-1).expand(
                -1, -1, tokens.shape[-1]
            )
            tokens = torch.gather(tokens, dim=1, index=idx)

        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)

        return tokens


class JEPAPredictor(nn.Module):
    """
    Predictor receives context tokens and target position queries.
    It predicts target embeddings.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        depth: int = 2,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.blocks = nn.TransformerEncoder(
            layer,
            num_layers=depth,
        )

        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        context_tokens: torch.Tensor,
        target_pos_embed: torch.Tensor,
    ) -> torch.Tensor:
        B, T, D = target_pos_embed.shape

        target_queries = self.mask_token.expand(B, T, D)
        target_queries = target_queries + target_pos_embed

        x = torch.cat([context_tokens, target_queries], dim=1)
        x = self.blocks(x)
        x = self.norm(x)

        pred_target_tokens = x[:, -T:, :]

        return pred_target_tokens


class TumorAwareJEPA(nn.Module):
    """
    JEPA model:

    context encoder:
        sees visible/context patches

    target encoder:
        EMA copy of context encoder
        sees full image and provides target embeddings

    predictor:
        predicts target embeddings from context embeddings
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 192,
        encoder_depth: int = 4,
        predictor_depth: int = 2,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.context_encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.target_encoder = copy.deepcopy(self.context_encoder)

        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = JEPAPredictor(
            embed_dim=embed_dim,
            depth=predictor_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.num_patches = self.context_encoder.num_patches
        self.embed_dim = embed_dim

    @torch.no_grad()
    def update_target_encoder(self, ema: float):
        """
        EMA update:
        target = ema * target + (1 - ema) * context
        """

        for target_param, context_param in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters(),
        ):
            target_param.data.mul_(ema).add_(
                context_param.data,
                alpha=1.0 - ema,
            )

    def _gather_pos_embed(self, indices: torch.Tensor) -> torch.Tensor:
        B = indices.shape[0]

        pos = self.context_encoder.pos_embed.expand(B, -1, -1)
        idx = indices.unsqueeze(-1).expand(-1, -1, pos.shape[-1])

        return torch.gather(pos, dim=1, index=idx)

    def forward(
        self,
        images: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ):
        # Context branch
        z_context = self.context_encoder(
            images,
            token_indices=context_indices,
        )

        # Target branch
        with torch.no_grad():
            z_all_target = self.target_encoder(images, token_indices=None)

            idx = target_indices.unsqueeze(-1).expand(
                -1, -1, z_all_target.shape[-1]
            )

            z_target = torch.gather(
                z_all_target,
                dim=1,
                index=idx,
            )

        # Predictor branch
        target_pos_embed = self._gather_pos_embed(target_indices)

        z_pred = self.predictor(
            context_tokens=z_context,
            target_pos_embed=target_pos_embed,
        )

        return z_pred, z_target