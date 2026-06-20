"""
Run inference on a single new CT volume (no ground-truth label required).

Usage:
    python -m src.predict --input path/to/volume.nii.gz --output path/to/prediction.nii.gz
    python -m src.predict --input path/to/volume.nii.gz --output pred.nii.gz --checkpoint checkpoints/best_model.pt
"""
import argparse

import nibabel as nib
import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Spacing, Orientation,
    ScaleIntensityRange, CropForeground, EnsureType,
)

from src import config
from src.engine import sliding_window_predict
from src.models.unet_3d import build_unet_3d


def get_predict_transform():
    """
    Mirrors get_val_transforms() in src/data/transforms.py but operates on
    a single (non-dictionary) image, since there's no label at inference time.
    """
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Spacing(pixdim=config.TARGET_SPACING, mode="bilinear"),
        Orientation(axcodes="RAS"),
        ScaleIntensityRange(
            a_min=config.INTENSITY_CLIP[0],
            a_max=config.INTENSITY_CLIP[1],
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForeground(),
        EnsureType(),
    ])


def parse_args():
    parser = argparse.ArgumentParser(description="Run spleen segmentation inference on a new volume")
    parser.add_argument("--input", type=str, required=True, help="Path to input .nii.gz CT volume")
    parser.add_argument("--output", type=str, required=True, help="Path to write predicted mask .nii.gz")
    parser.add_argument(
        "--checkpoint", type=str, default=str(config.CHECKPOINT_DIR / "best_model.pt")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_unet_3d().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loading and preprocessing {args.input}...")
    transform = get_predict_transform()
    image: MetaTensor = transform(args.input)
    image_batch = image.unsqueeze(0).to(device)  # add batch dim -> (1, C, H, W, D)

    print("Running sliding-window inference...")
    with torch.no_grad():
        logits = sliding_window_predict(model, image_batch)
        pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Recover affine from the preprocessed image's metadata so the output
    # mask is spatially aligned with the (resampled) input volume.
    affine = image.affine.numpy() if hasattr(image, "affine") else np.eye(4)
    out_img = nib.Nifti1Image(pred_mask, affine)
    nib.save(out_img, args.output)

    voxel_count = int(pred_mask.sum())
    print(f"Saved predicted mask to {args.output}")
    print(f"Predicted spleen volume: {voxel_count} voxels at {config.TARGET_SPACING} mm spacing")
    print(
        "NOTE: this mask is in the *resampled* (1.5mm isotropic, foreground-cropped) "
        "space used for inference, not the original volume's native space/size. "
        "For clinical-grade output you'd resample the prediction back onto the "
        "original image grid -- left as a known next step, see README."
    )


if __name__ == "__main__":
    main()
