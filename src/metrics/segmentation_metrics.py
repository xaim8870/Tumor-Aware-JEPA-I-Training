# src/metrics/segmentation_metrics.py

import numpy as np
import torch


@torch.no_grad()
def batch_confusion_counts_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
):
    """
    Pixel-wise binary confusion counts.

    Returns:
        tn, fp, fn, tp
    """

    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).bool()
    targets = (targets >= 0.5).bool()

    preds = preds.view(-1)
    targets = targets.view(-1)

    tp = torch.logical_and(preds == 1, targets == 1).sum().item()
    tn = torch.logical_and(preds == 0, targets == 0).sum().item()
    fp = torch.logical_and(preds == 1, targets == 0).sum().item()
    fn = torch.logical_and(preds == 0, targets == 1).sum().item()

    return tn, fp, fn, tp


def metrics_from_confusion(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    eps: float = 1e-7,
):
    """
    Computes binary segmentation metrics from pixel-wise confusion counts.
    """

    total = tn + fp + fn + tp

    accuracy = (tp + tn) / (total + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)

    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)

    f1 = dice

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "dice": dice,
        "iou": iou,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def confusion_matrix_array(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
):
    """
    Returns confusion matrix in this format:

        [[TN, FP],
         [FN, TP]]
    """

    return np.array(
        [
            [tn, fp],
            [fn, tp],
        ],
        dtype=np.int64,
    )