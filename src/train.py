"""
Train the 3D U-Net on MSD Task09_Spleen.

Usage:
    python -m src.train
    python -m src.train --epochs 50 --no-wandb

Designed to run inside a Colab/Kaggle notebook (see notebooks/01_train.ipynb)
as much as from the command line -- set the SPLEEN_DATA_DIR / SPLEEN_CKPT_DIR
/ SPLEEN_REPORTS_DIR env vars before importing src.config to point at your
environment's storage (e.g. a Drive-mounted folder on Colab).
"""
import argparse
import json
import time
from pathlib import Path

import torch

from src import config
from src.data.msd_spleen_datamodule import get_dataloaders
from src.engine import evaluate_loader, get_dice_metric, get_loss_fn, get_postprocessing
from src.models.unet_3d import build_unet_3d


def parse_args():
    parser = argparse.ArgumentParser(description="Train spleen segmentation U-Net")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--cache-rate", type=float, default=1.0)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print(
            "WARNING: training a 3D U-Net on CPU will be very slow. "
            "Make sure your Colab/Kaggle runtime has a GPU attached."
        )

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            name=args.run_name,
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "patch_size": config.PATCH_SIZE,
                "target_spacing": config.TARGET_SPACING,
                "intensity_clip": config.INTENSITY_CLIP,
                "architecture": "MONAI UNet (16,32,64,128,256), 2 res units/block",
            },
        )

    print("Building dataloaders (first run will cache-preprocess all volumes, "
          "this can take a few minutes)...")
    train_loader, val_loader, test_loader = get_dataloaders(cache_rate=args.cache_rate)
    print(f"train={len(train_loader.dataset)} cases, "
          f"val={len(val_loader.dataset)} cases, "
          f"test={len(test_loader.dataset)} cases (held out, untouched until evaluate.py)")

    model = build_unet_3d().to(device)
    loss_fn = get_loss_fn()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    post_pred, post_label = get_postprocessing()

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = config.CHECKPOINT_DIR / "best_model.pt"

    best_val_dice = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        epoch_time = time.time() - epoch_start

        log_payload = {"epoch": epoch, "train_loss": train_loss, "epoch_time_sec": epoch_time}
        print(f"[epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f} ({epoch_time:.1f}s)")

        if epoch % config.VAL_INTERVAL == 0:
            val_dice, _ = evaluate_loader(model, val_loader, device)
            log_payload["val_dice"] = val_dice
            print(f"  -> val_dice={val_dice:.4f}")

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                epochs_without_improvement = 0
                torch.save(
                    {"model_state_dict": model.state_dict(), "epoch": epoch, "val_dice": val_dice},
                    best_ckpt_path,
                )
                print(f"  -> new best model saved (val_dice={val_dice:.4f}) -> {best_ckpt_path}")
            else:
                epochs_without_improvement += 1

        if use_wandb:
            wandb.log(log_payload)
        history.append(log_payload)

        if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
            print(
                f"No val_dice improvement for {config.EARLY_STOPPING_PATIENCE} "
                f"validation checks -- stopping early at epoch {epoch}."
            )
            break

    history_path = config.REPORTS_DIR / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"Training history saved to {history_path}")
    print(f"Best val Dice: {best_val_dice:.4f} (checkpoint: {best_ckpt_path})")

    if use_wandb:
        wandb.summary["best_val_dice"] = best_val_dice
        wandb.finish()


if __name__ == "__main__":
    main()
