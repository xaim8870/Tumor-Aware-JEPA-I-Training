# src/models/masking.py

import torch
import torch.nn.functional as F


def _minmax_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize each sample independently to [0, 1].
    x shape: [B, N]
    """
    x_min = x.min(dim=1, keepdim=True).values
    x_max = x.max(dim=1, keepdim=True).values
    return (x - x_min) / (x_max - x_min + eps)


def compute_mri_patch_saliency(
    images: torch.Tensor,
    patch_size: int = 16,
) -> torch.Tensor:
    """
    Computes an unsupervised MRI saliency score for each patch.

    images: [B, 1, H, W], normalized roughly in [-1, 1]

    Output:
        scores: [B, N_patches]

    This is our first tumor-aware idea:
    patches with unusual intensity, strong edges, and high texture variance
    are more likely to be selected as JEPA target regions.
    """

    B, C, H, W = images.shape
    assert C == 1, "This version expects grayscale MRI images."
    assert H % patch_size == 0 and W % patch_size == 0

    # Convert from [-1, 1] to [0, 1]
    x = (images + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)

    # -----------------------------
    # 1. Intensity abnormality score
    # -----------------------------
    global_mean = x.mean(dim=(2, 3), keepdim=True)

    patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)
    # [B, patch_size*patch_size, N]
    patches = patches.transpose(1, 2)
    # [B, N, patch_pixels]

    patch_mean = patches.mean(dim=2)
    patch_var = patches.var(dim=2)

    intensity_score = (patch_mean - global_mean.view(B, 1)).abs()

    # -----------------------------
    # 2. Texture score
    # -----------------------------
    texture_score = patch_var

    # -----------------------------
    # 3. Edge score using Sobel filters
    # -----------------------------
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)

    grad_x = F.conv2d(x, sobel_x, padding=1)
    grad_y = F.conv2d(x, sobel_y, padding=1)

    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

    grad_patches = F.unfold(grad_mag, kernel_size=patch_size, stride=patch_size)
    grad_patches = grad_patches.transpose(1, 2)
    edge_score = grad_patches.mean(dim=2)

    # Normalize each score type
    intensity_score = _minmax_norm(intensity_score)
    texture_score = _minmax_norm(texture_score)
    edge_score = _minmax_norm(edge_score)

    # Combined tumor-aware / MRI-saliency score
    scores = (
        0.45 * intensity_score
        + 0.35 * edge_score
        + 0.20 * texture_score
    )

    scores = _minmax_norm(scores)

    return scores


def sample_target_indices_from_scores(
    scores: torch.Tensor,
    target_ratio: float = 0.25,
    alpha: float = 4.0,
) -> torch.Tensor:
    """
    Samples target patch indices using saliency scores.

    scores: [B, N]

    Higher score = higher chance of being selected as target.
    """

    B, N = scores.shape
    num_targets = max(1, int(N * target_ratio))

    probs = torch.softmax(alpha * scores, dim=1)

    target_indices = torch.multinomial(
        probs,
        num_samples=num_targets,
        replacement=False,
    )

    target_indices, _ = torch.sort(target_indices, dim=1)

    return target_indices


def sample_random_target_indices(
    batch_size: int,
    num_patches: int,
    target_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Standard random JEPA masking baseline.
    """

    num_targets = max(1, int(num_patches * target_ratio))

    all_indices = []
    for _ in range(batch_size):
        perm = torch.randperm(num_patches, device=device)
        idx = perm[:num_targets]
        idx, _ = torch.sort(idx)
        all_indices.append(idx)

    return torch.stack(all_indices, dim=0)


def make_context_indices(
    target_indices: torch.Tensor,
    num_patches: int,
) -> torch.Tensor:
    """
    Given target indices, returns all remaining patch indices as context.

    target_indices: [B, T]
    output: [B, N-T]
    """

    B = target_indices.shape[0]
    device = target_indices.device

    context_indices = []

    for b in range(B):
        mask = torch.ones(num_patches, dtype=torch.bool, device=device)
        mask[target_indices[b]] = False
        ctx = torch.arange(num_patches, device=device)[mask]
        context_indices.append(ctx)

    return torch.stack(context_indices, dim=0)