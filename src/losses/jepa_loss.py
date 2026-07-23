# src/losses/jepa_loss.py

import torch
import torch.nn.functional as F


def jepa_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    MSE loss between predicted target embeddings and actual target embeddings.
    """

    return F.mse_loss(pred, target)


def jepa_cosine_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Cosine distance loss.
    """

    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)

    cosine_sim = (pred * target).sum(dim=-1)

    return 1.0 - cosine_sim.mean()


def jepa_combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mse_weight: float = 1.0,
    cosine_weight: float = 0.1,
) -> torch.Tensor:
    """
    Combined embedding prediction loss.
    """

    mse = jepa_mse_loss(pred, target)
    cos = jepa_cosine_loss(pred, target)

    return mse_weight * mse + cosine_weight * cos