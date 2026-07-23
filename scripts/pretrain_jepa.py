# scripts/pretrain_jepa.py

import sys
import math
import random
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow imports from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.datasets.brisc_jepa_dataset import BRISCJEPADataset
from src.models.jepa_vit import TumorAwareJEPA
from src.models.masking import (
    compute_mri_patch_saliency,
    sample_target_indices_from_scores,
    sample_random_target_indices,
    make_context_indices,
)
from src.losses.jepa_loss import jepa_combined_loss


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_ema_schedule(
    epoch: int,
    total_epochs: int,
    ema_start: float,
    ema_end: float,
) -> float:
    """
    Smoothly increases EMA value from ema_start to ema_end.
    """

    if total_epochs <= 1:
        return ema_end

    t = epoch / (total_epochs - 1)
    cosine = 0.5 * (1.0 - math.cos(math.pi * t))

    return ema_start + cosine * (ema_end - ema_start)


def get_image_dir(data_root: Path, split_path: str) -> Path:
    image_dir = data_root / split_path

    if not image_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {image_dir}")

    return image_dir


def move_optimizer_to_device(optimizer, device):
    """
    Useful when optimizer state is loaded from checkpoint.
    Ensures optimizer tensors are on the current device.
    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    cfg,
    epoch,
):
    model.train()

    total_loss = 0.0
    num_batches = 0

    patch_size = cfg["model"]["patch_size"]
    target_ratio = cfg["masking"]["target_ratio"]
    masking_strategy = cfg["masking"]["strategy"]
    saliency_alpha = cfg["masking"]["saliency_alpha"]
    use_amp = cfg["training"]["use_amp"]
    grad_clip = cfg["training"]["grad_clip"]

    ema = cosine_ema_schedule(
        epoch=epoch,
        total_epochs=cfg["training"]["epochs"],
        ema_start=cfg["training"]["ema_start"],
        ema_end=cfg["training"]["ema_end"],
    )

    pbar = tqdm(loader, desc=f"Train Epoch {epoch + 1}", leave=False)

    for images, _names in pbar:
        images = images.to(device, non_blocking=True)
        B = images.shape[0]
        N = model.num_patches

        if masking_strategy == "tumor_aware":
            scores = compute_mri_patch_saliency(
                images,
                patch_size=patch_size,
            )

            target_indices = sample_target_indices_from_scores(
                scores=scores,
                target_ratio=target_ratio,
                alpha=saliency_alpha,
            )

        elif masking_strategy == "random":
            target_indices = sample_random_target_indices(
                batch_size=B,
                num_patches=N,
                target_ratio=target_ratio,
                device=device,
            )

        else:
            raise ValueError(f"Unknown masking strategy: {masking_strategy}")

        context_indices = make_context_indices(
            target_indices=target_indices,
            num_patches=N,
        )

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                z_pred, z_target = model(
                    images=images,
                    context_indices=context_indices,
                    target_indices=target_indices,
                )

                loss = jepa_combined_loss(z_pred, z_target)

            scaler.scale(loss).backward()

            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )

            scaler.step(optimizer)
            scaler.update()

        else:
            z_pred, z_target = model(
                images=images,
                context_indices=context_indices,
                target_indices=target_indices,
            )

            loss = jepa_combined_loss(z_pred, z_target)
            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )

            optimizer.step()

        model.update_target_encoder(ema=ema)

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.5f}",
                "ema": f"{ema:.6f}",
            }
        )

    avg_loss = total_loss / max(1, num_batches)

    return avg_loss, ema


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    device,
    cfg,
):
    model.eval()

    total_loss = 0.0
    num_batches = 0

    patch_size = cfg["model"]["patch_size"]
    target_ratio = cfg["masking"]["target_ratio"]
    masking_strategy = cfg["masking"]["strategy"]
    saliency_alpha = cfg["masking"]["saliency_alpha"]

    pbar = tqdm(loader, desc="Val", leave=False)

    for images, _names in pbar:
        images = images.to(device, non_blocking=True)
        B = images.shape[0]
        N = model.num_patches

        if masking_strategy == "tumor_aware":
            scores = compute_mri_patch_saliency(
                images,
                patch_size=patch_size,
            )

            target_indices = sample_target_indices_from_scores(
                scores=scores,
                target_ratio=target_ratio,
                alpha=saliency_alpha,
            )

        elif masking_strategy == "random":
            target_indices = sample_random_target_indices(
                batch_size=B,
                num_patches=N,
                target_ratio=target_ratio,
                device=device,
            )

        else:
            raise ValueError(f"Unknown masking strategy: {masking_strategy}")

        context_indices = make_context_indices(
            target_indices=target_indices,
            num_patches=N,
        )

        z_pred, z_target = model(
            images=images,
            context_indices=context_indices,
            target_indices=target_indices,
        )

        loss = jepa_combined_loss(z_pred, z_target)

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.5f}"})

    avg_loss = total_loss / max(1, num_batches)

    return avg_loss


def main():
    config_path = ROOT / "configs" / "pretrain_jepa.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("Tumor-Aware JEPA Pretraining")
    print("=" * 70)
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_root = Path(cfg["data"]["data_root"])
    train_dir = get_image_dir(data_root, cfg["data"]["train_images"])
    val_dir = get_image_dir(data_root, cfg["data"]["val_images"])

    print(f"Train images: {train_dir}")
    print(f"Val images:   {val_dir}")

    train_dataset = BRISCJEPADataset(
        train_dir,
        img_size=cfg["data"]["img_size"],
    )

    val_dataset = BRISCJEPADataset(
        val_dir,
        img_size=cfg["data"]["img_size"],
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    model = TumorAwareJEPA(
        img_size=cfg["model"]["img_size"],
        patch_size=cfg["model"]["patch_size"],
        in_channels=cfg["model"]["in_channels"],
        embed_dim=cfg["model"]["embed_dim"],
        encoder_depth=cfg["model"]["encoder_depth"],
        predictor_depth=cfg["model"]["predictor_depth"],
        num_heads=cfg["model"]["num_heads"],
        mlp_ratio=cfg["model"]["mlp_ratio"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    print("\nModel created.")
    print(f"Number of patches: {model.num_patches}")
    print(f"Embedding dim:     {model.embed_dim}")
    print(f"Masking strategy:  {cfg['masking']['strategy']}")
    print(f"Target ratio:      {cfg['masking']['target_ratio']}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(cfg["training"]["use_amp"] and device.type == "cuda")
    )

    out_dir = ROOT / cfg["output"]["out_dir"]
    ckpt_dir = out_dir / "checkpoints"
    csv_dir = out_dir / "csv"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history = []
    start_epoch = 0

    resume_from = cfg["training"].get("resume_from", None)

    if resume_from:
        resume_path = ROOT / resume_from

        if resume_path.exists():
            print(f"\nResuming from checkpoint: {resume_path}")

            checkpoint = torch.load(
                resume_path,
                map_location=device,
                weights_only=False,
            )

            model.load_state_dict(checkpoint["model_state_dict"])

            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                move_optimizer_to_device(optimizer, device)

            if "scaler_state_dict" in checkpoint and scaler is not None:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])

            start_epoch = int(checkpoint.get("epoch", 0))

            old_csv_path = csv_dir / "pretrain_jepa_metrics.csv"

            if old_csv_path.exists():
                old_df = pd.read_csv(old_csv_path)

                # Keep only rows up to the checkpoint epoch
                # This avoids duplicate rows if you resume multiple times.
                old_df = old_df[old_df["epoch"] <= start_epoch]

                history = old_df.to_dict("records")

                if len(old_df) > 0:
                    best_val_loss = float(old_df["val_loss"].min())

            print(f"Resume successful.")
            print(f"Checkpoint epoch: {start_epoch}")
            print(f"Starting from epoch {start_epoch + 1}.")
            print(f"Best previous val loss: {best_val_loss:.6f}")

        else:
            print(f"\nResume checkpoint not found: {resume_path}")
            print("Starting from scratch.")

    epochs = cfg["training"]["epochs"]

    if start_epoch >= epochs:
        print(
            f"\nCheckpoint is already at epoch {start_epoch}, "
            f"but config epochs is {epochs}."
        )
        print("Nothing to train. Increase epochs in config if you want to continue.")

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch [{epoch + 1}/{epochs}]")

        train_loss, ema = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            cfg=cfg,
            epoch=epoch,
        )

        val_loss = validate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            cfg=cfg,
        )

        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss:   {val_loss:.6f}")
        print(f"EMA:        {ema:.6f}")

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "ema": ema,
            "masking_strategy": cfg["masking"]["strategy"],
            "target_ratio": cfg["masking"]["target_ratio"],
            "img_size": cfg["model"]["img_size"],
            "patch_size": cfg["model"]["patch_size"],
            "embed_dim": cfg["model"]["embed_dim"],
        }

        history.append(row)

        history_df = pd.DataFrame(history)
        history_df.to_csv(csv_dir / "pretrain_jepa_metrics.csv", index=False)

        last_ckpt = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "context_encoder_state_dict": model.context_encoder.state_dict(),
            "target_encoder_state_dict": model.target_encoder.state_dict(),
            "predictor_state_dict": model.predictor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "cfg": cfg,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        torch.save(last_ckpt, ckpt_dir / "last_jepa_checkpoint.pth")

        # Keep the mathematically best val-loss checkpoint.
        # But for JEPA representation learning, we will also save final encoder below.
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(last_ckpt, ckpt_dir / "best_jepa_checkpoint.pth")

            torch.save(
                model.context_encoder.state_dict(),
                ckpt_dir / "best_jepa_encoder_only.pth",
            )

            print("Best JEPA encoder saved.")

    final_epoch = max(start_epoch, epochs)

    final_encoder_path = ckpt_dir / f"final_jepa_encoder_{final_epoch}ep.pth"

    torch.save(
        model.context_encoder.state_dict(),
        final_encoder_path,
    )

    final_full_ckpt_path = ckpt_dir / f"final_jepa_checkpoint_{final_epoch}ep.pth"

    final_ckpt = {
        "epoch": final_epoch,
        "model_state_dict": model.state_dict(),
        "context_encoder_state_dict": model.context_encoder.state_dict(),
        "target_encoder_state_dict": model.target_encoder.state_dict(),
        "predictor_state_dict": model.predictor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "cfg": cfg,
        "best_val_loss": best_val_loss,
    }

    torch.save(final_ckpt, final_full_ckpt_path)

    print("\nTraining complete.")
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Final JEPA encoder saved to: {final_encoder_path}")
    print(f"Final full checkpoint saved to: {final_full_ckpt_path}")
    print(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()