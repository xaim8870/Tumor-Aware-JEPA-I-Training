# src/losses/segmentation_losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Dice loss for binary segmentation.

    logits:  [B, 1, H, W]
    targets: [B, 1, H, W]
    """

    probs = torch.sigmoid(logits)

    dims = (1, 2, 3)

    intersection = torch.sum(probs * targets, dim=dims)
    union = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)

    dice = (2.0 * intersection + eps) / (union + eps)

    return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    BCEWithLogits + Dice loss.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ):
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        bce = self.bce(logits, targets)
        dice = dice_loss_from_logits(logits, targets)

        return self.bce_weight * bce + self.dice_weight * dice