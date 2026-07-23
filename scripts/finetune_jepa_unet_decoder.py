# scripts/finetune_jepa_unet_decoder.py

import sys
import random
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Allow imports from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.datasets.brisc_seg_dataset import BRISCSegmentationDataset
from src.models.jepa_unet_segmenter import JEPAUNetSegmenter
from src.losses.segmentation_losses import BCEDiceLoss
from src.metrics.segmentation_metrics import (
    batch_confusion_counts_from_logits,
    metrics_from_confusion,
    confusion_matrix_array,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dir(data_root: Path, split_path: str) -> Path:
    path = data_root / split_path

    if not path.exists():
        raise FileNotFoundError(f"Folder not found: {path}")

    return path


def set_encoder_trainable(model: JEPAUNetSegmenter, trainable: bool):
    for p in model.encoder.parameters():
        p.requires_grad = trainable


def create_optimizer(model, cfg):
    encoder_lr = cfg["training"]["encoder_lr"]
    decoder_lr = cfg["training"]["decoder_lr"]
    weight_decay = cfg["training"]["weight_decay"]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": encoder_lr,
            },
            {
                "params": model.decoder.parameters(),
                "lr": decoder_lr,
            },
        ],
        weight_decay=weight_decay,
    )

    return optimizer


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    cfg,
    epoch,
):
    model.train()

    freeze_encoder_epochs = cfg["training"]["freeze_encoder_epochs"]

    if epoch < freeze_encoder_epochs:
        set_encoder_trainable(model, False)
    else:
        set_encoder_trainable(model, True)

    total_loss = 0.0
    num_batches = 0

    tn_total = 0
    fp_total = 0
    fn_total = 0
    tp_total = 0

    use_amp = cfg["training"]["use_amp"]
    grad_clip = cfg["training"]["grad_clip"]
    threshold = cfg["training"]["threshold"]

    pbar = tqdm(loader, desc=f"Train Epoch {epoch + 1}", leave=False)

    for images, masks, _names in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, masks)

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
            logits = model(images)
            loss = criterion(logits, masks)

            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )

            optimizer.step()

        tn, fp, fn, tp = batch_confusion_counts_from_logits(
            logits=logits.detach(),
            targets=masks.detach(),
            threshold=threshold,
        )

        tn_total += tn
        fp_total += fp
        fn_total += fn
        tp_total += tp

        total_loss += loss.item()
        num_batches += 1

        running_metrics = metrics_from_confusion(
            tn_total,
            fp_total,
            fn_total,
            tp_total,
        )

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "dice": f"{running_metrics['dice']:.4f}",
                "iou": f"{running_metrics['iou']:.4f}",
            }
        )

    avg_loss = total_loss / max(1, num_batches)

    metrics = metrics_from_confusion(
        tn_total,
        fp_total,
        fn_total,
        tp_total,
    )

    metrics["loss"] = avg_loss

    return metrics


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
    cfg,
):
    model.eval()

    total_loss = 0.0
    num_batches = 0

    tn_total = 0
    fp_total = 0
    fn_total = 0
    tp_total = 0

    threshold = cfg["training"]["threshold"]

    pbar = tqdm(loader, desc="Val", leave=False)

    for images, masks, _names in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)

        loss = criterion(logits, masks)

        tn, fp, fn, tp = batch_confusion_counts_from_logits(
            logits=logits,
            targets=masks,
            threshold=threshold,
        )

        tn_total += tn
        fp_total += fp
        fn_total += fn
        tp_total += tp

        total_loss += loss.item()
        num_batches += 1

        running_metrics = metrics_from_confusion(
            tn_total,
            fp_total,
            fn_total,
            tp_total,
        )

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "dice": f"{running_metrics['dice']:.4f}",
                "iou": f"{running_metrics['iou']:.4f}",
            }
        )

    avg_loss = total_loss / max(1, num_batches)

    metrics = metrics_from_confusion(
        tn_total,
        fp_total,
        fn_total,
        tp_total,
    )

    metrics["loss"] = avg_loss

    return metrics


@torch.no_grad()
def final_validation_confusion(
    model,
    loader,
    device,
    cfg,
):
    model.eval()

    threshold = cfg["training"]["threshold"]

    tn_total = 0
    fp_total = 0
    fn_total = 0
    tp_total = 0

    for images, masks, _names in tqdm(loader, desc="Final Confusion Matrix", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)

        tn, fp, fn, tp = batch_confusion_counts_from_logits(
            logits=logits,
            targets=masks,
            threshold=threshold,
        )

        tn_total += tn
        fp_total += fp
        fn_total += fn
        tp_total += tp

    metrics = metrics_from_confusion(
        tn_total,
        fp_total,
        fn_total,
        tp_total,
    )

    cm = confusion_matrix_array(
        tn_total,
        fp_total,
        fn_total,
        tp_total,
    )

    return cm, metrics


def save_confusion_matrix_outputs(
    cm,
    metrics,
    csv_dir: Path,
    plot_dir: Path,
):
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Background", "Actual Tumor"],
        columns=["Pred Background", "Pred Tumor"],
    )

    cm_df.to_csv(csv_dir / "final_val_confusion_matrix.csv")

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(csv_dir / "final_val_confusion_metrics.csv", index=False)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Final Validation Pixel-wise Confusion Matrix")
    plt.xticks([0, 1], ["Pred BG", "Pred Tumor"])
    plt.yticks([0, 1], ["Actual BG", "Actual Tumor"])

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(plot_dir / "final_val_confusion_matrix.png", dpi=200)
    plt.close()


def save_training_plots(
    history_df: pd.DataFrame,
    plot_dir: Path,
):
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("JEPA + U-Net Decoder Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_dice"], label="Train Dice")
    plt.plot(history_df["epoch"], history_df["val_dice"], label="Val Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("JEPA + U-Net Decoder Dice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "dice_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_iou"], label="Train IoU")
    plt.plot(history_df["epoch"], history_df["val_iou"], label="Val IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("JEPA + U-Net Decoder IoU")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "iou_curve.png", dpi=200)
    plt.close()


def main():
    config_path = ROOT / "configs" / "finetune_jepa_unet.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Fine-tuning JEPA Encoder + U-Net Decoder for Brain Tumor Segmentation")
    print("=" * 80)
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_root = Path(cfg["data"]["data_root"])

    train_img_dir = get_dir(data_root, cfg["data"]["train_images"])
    train_mask_dir = get_dir(data_root, cfg["data"]["train_masks"])

    val_img_dir = get_dir(data_root, cfg["data"]["val_images"])
    val_mask_dir = get_dir(data_root, cfg["data"]["val_masks"])

    print(f"Train images: {train_img_dir}")
    print(f"Train masks:  {train_mask_dir}")
    print(f"Val images:   {val_img_dir}")
    print(f"Val masks:    {val_mask_dir}")

    train_dataset = BRISCSegmentationDataset(
        image_dir=train_img_dir,
        mask_dir=train_mask_dir,
        img_size=cfg["data"]["img_size"],
        mask_threshold=cfg["data"]["mask_threshold"],
    )

    val_dataset = BRISCSegmentationDataset(
        image_dir=val_img_dir,
        mask_dir=val_mask_dir,
        img_size=cfg["data"]["img_size"],
        mask_threshold=cfg["data"]["mask_threshold"],
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

    model = JEPAUNetSegmenter(
        img_size=cfg["model"]["img_size"],
        patch_size=cfg["model"]["patch_size"],
        in_channels=cfg["model"]["in_channels"],
        embed_dim=cfg["model"]["embed_dim"],
        encoder_depth=cfg["model"]["encoder_depth"],
        num_heads=cfg["model"]["num_heads"],
        mlp_ratio=cfg["model"]["mlp_ratio"],
        dropout=cfg["model"]["dropout"],
        decoder_channels=cfg["model"]["decoder_channels"],
    )

    encoder_ckpt = ROOT / cfg["pretrained"]["encoder_checkpoint"]
    model.load_jepa_encoder(str(encoder_ckpt))

    model = model.to(device)

    criterion = BCEDiceLoss(
        bce_weight=cfg["training"]["loss_bce_weight"],
        dice_weight=cfg["training"]["loss_dice_weight"],
    )

    optimizer = create_optimizer(model, cfg)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(cfg["training"]["use_amp"] and device.type == "cuda")
    )

    out_dir = ROOT / cfg["output"]["out_dir"]
    ckpt_dir = out_dir / "checkpoints"
    csv_dir = out_dir / "csv"
    plot_dir = out_dir / "plots"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    best_val_dice = -1.0
    best_val_iou = -1.0

    history = []

    epochs = cfg["training"]["epochs"]

    for epoch in range(epochs):
        print(f"\nEpoch [{epoch + 1}/{epochs}]")

        if epoch < cfg["training"]["freeze_encoder_epochs"]:
            print("Encoder: frozen")
        else:
            print("Encoder: trainable")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            cfg=cfg,
            epoch=epoch,
        )

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            cfg=cfg,
        )

        scheduler.step()

        encoder_lr = optimizer.param_groups[0]["lr"]
        decoder_lr = optimizer.param_groups[1]["lr"]

        print(f"Train Loss: {train_metrics['loss']:.6f}")
        print(f"Val Loss:   {val_metrics['loss']:.6f}")

        print(f"Train Dice: {train_metrics['dice']:.6f}")
        print(f"Val Dice:   {val_metrics['dice']:.6f}")

        print(f"Train IoU:  {train_metrics['iou']:.6f}")
        print(f"Val IoU:    {val_metrics['iou']:.6f}")

        print(f"Val Precision: {val_metrics['precision']:.6f}")
        print(f"Val Recall:    {val_metrics['recall']:.6f}")

        row = {
            "epoch": epoch + 1,

            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_specificity": train_metrics["specificity"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "train_f1": train_metrics["f1"],
            "train_tn": train_metrics["tn"],
            "train_fp": train_metrics["fp"],
            "train_fn": train_metrics["fn"],
            "train_tp": train_metrics["tp"],

            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_specificity": val_metrics["specificity"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_f1": val_metrics["f1"],
            "val_tn": val_metrics["tn"],
            "val_fp": val_metrics["fp"],
            "val_fn": val_metrics["fn"],
            "val_tp": val_metrics["tp"],

            "encoder_lr": encoder_lr,
            "decoder_lr": decoder_lr,
            "encoder_frozen": epoch < cfg["training"]["freeze_encoder_epochs"],
        }

        history.append(row)

        history_df = pd.DataFrame(history)
        history_df.to_csv(csv_dir / "finetune_jepa_unet_metrics.csv", index=False)

        last_ckpt = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "decoder_state_dict": model.decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "cfg": cfg,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }

        torch.save(last_ckpt, ckpt_dir / "last_jepa_unet_segmentation.pth")

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_val_iou = val_metrics["iou"]

            torch.save(last_ckpt, ckpt_dir / "best_jepa_unet_segmentation.pth")

            print("Best segmentation checkpoint saved.")

    history_df = pd.DataFrame(history)
    history_df.to_csv(csv_dir / "finetune_jepa_unet_metrics.csv", index=False)

    save_training_plots(
        history_df=history_df,
        plot_dir=plot_dir,
    )

    print("\nGenerating final validation confusion matrix using best checkpoint...")

    best_ckpt_path = ckpt_dir / "best_jepa_unet_segmentation.pth"

    if best_ckpt_path.exists():
        try:
            best_ckpt = torch.load(
                best_ckpt_path,
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            best_ckpt = torch.load(
                best_ckpt_path,
                map_location=device,
            )

        model.load_state_dict(best_ckpt["model_state_dict"])
        model = model.to(device)

    cm, final_metrics = final_validation_confusion(
        model=model,
        loader=val_loader,
        device=device,
        cfg=cfg,
    )

    save_confusion_matrix_outputs(
        cm=cm,
        metrics=final_metrics,
        csv_dir=csv_dir,
        plot_dir=plot_dir,
    )

    final_summary = {
        "best_val_dice": best_val_dice,
        "best_val_iou": best_val_iou,
        "final_confusion_tn": final_metrics["tn"],
        "final_confusion_fp": final_metrics["fp"],
        "final_confusion_fn": final_metrics["fn"],
        "final_confusion_tp": final_metrics["tp"],
        "final_accuracy": final_metrics["accuracy"],
        "final_precision": final_metrics["precision"],
        "final_recall": final_metrics["recall"],
        "final_specificity": final_metrics["specificity"],
        "final_dice": final_metrics["dice"],
        "final_iou": final_metrics["iou"],
        "final_f1": final_metrics["f1"],
    }

    pd.DataFrame([final_summary]).to_csv(
        csv_dir / "final_summary.csv",
        index=False,
    )

    print("\nTraining complete.")
    print(f"Best Val Dice: {best_val_dice:.6f}")
    print(f"Best Val IoU:  {best_val_iou:.6f}")
    print(f"Final Dice:    {final_metrics['dice']:.6f}")
    print(f"Final IoU:     {final_metrics['iou']:.6f}")
    print(f"Final Precision: {final_metrics['precision']:.6f}")
    print(f"Final Recall:    {final_metrics['recall']:.6f}")

    print("\nSaved files:")
    print(f"Metrics CSV:        {csv_dir / 'finetune_jepa_unet_metrics.csv'}")
    print(f"Final summary CSV:  {csv_dir / 'final_summary.csv'}")
    print(f"Confusion CSV:      {csv_dir / 'final_val_confusion_matrix.csv'}")
    print(f"Confusion plot:     {plot_dir / 'final_val_confusion_matrix.png'}")
    print(f"Best checkpoint:    {ckpt_dir / 'best_jepa_unet_segmentation.pth'}")
    print(f"Last checkpoint:    {ckpt_dir / 'last_jepa_unet_segmentation.pth'}")


if __name__ == "__main__":
    main()