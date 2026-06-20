"""
Evaluate a trained checkpoint on the held-out test set.

This is intentionally a separate script from train.py and uses the test
split (never seen during training or for checkpoint selection) so the
reported numbers are an honest estimate of generalization, not a number
that was implicitly selected for by "pick the epoch with best val Dice."

Usage:
    python -m src.evaluate
    python -m src.evaluate --checkpoint checkpoints/best_model.pt --n-worst 5
"""
import argparse
import json

import torch

from src import config
from src.data.msd_spleen_datamodule import get_dataloaders
from src.engine import evaluate_loader
from src.models.unet_3d import build_unet_3d


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate spleen segmentation model on test set")
    parser.add_argument(
        "--checkpoint", type=str, default=str(config.CHECKPOINT_DIR / "best_model.pt")
    )
    parser.add_argument(
        "--n-worst", type=int, default=5,
        help="Number of worst-performing cases to flag for qualitative review"
    )
    parser.add_argument("--cache-rate", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_unet_3d().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"Checkpoint from epoch {checkpoint.get('epoch', '?')}, "
        f"val_dice at save time = {checkpoint.get('val_dice', float('nan')):.4f}"
    )

    print("Building dataloaders...")
    _, _, test_loader = get_dataloaders(cache_rate=args.cache_rate)
    print(f"Evaluating on {len(test_loader.dataset)} held-out test cases (not used in training/val)")

    mean_dice, per_case = evaluate_loader(model, test_loader, device, return_per_case=True)

    sorted_cases = sorted(per_case.items(), key=lambda kv: kv[1])
    worst_cases = sorted_cases[: args.n_worst]
    best_cases = sorted_cases[-args.n_worst:]

    print(f"\n=== Test set results ===")
    print(f"Mean Dice (test, n={len(per_case)}): {mean_dice:.4f}")
    print(f"\nWorst {args.n_worst} cases (candidates for qualitative failure analysis):")
    for case_id, dice in worst_cases:
        print(f"  {case_id}: Dice={dice:.4f}")
    print(f"\nBest {args.n_worst} cases:")
    for case_id, dice in best_cases:
        print(f"  {case_id}: Dice={dice:.4f}")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "mean_test_dice": mean_dice,
        "n_test_cases": len(per_case),
        "per_case_dice": per_case,
        "worst_cases": worst_cases,
        "best_cases": best_cases,
    }
    results_path = config.REPORTS_DIR / "test_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nFull results saved to {results_path}")
    print(
        "Tip: visualize the worst cases (e.g. in a notebook) before claiming a "
        "final number -- low Dice on a tiny/edge-of-volume spleen often looks "
        "worse than it is, since Dice is harsh on small structures."
    )


if __name__ == "__main__":
    main()
