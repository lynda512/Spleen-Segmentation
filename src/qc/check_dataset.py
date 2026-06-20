"""
Basic dataset QC: confirms image/label pairing, and summarizes per-case
shape, voxel spacing, and intensity range so dataset assumptions baked
into config.py (e.g. TARGET_SPACING, INTENSITY_CLIP) are verifiable
rather than just guessed.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np

from src import config


def get_nifti_paths(root: Path) -> Tuple[List[Path], List[Path]]:
    def valid(p: Path) -> bool:
        return p.name.endswith(".nii.gz") and not p.name.startswith("._")

    images_tr = sorted([p for p in (root / "imagesTr").glob("*.nii.gz") if valid(p)])
    labels_tr = sorted([p for p in (root / "labelsTr").glob("*.nii.gz") if valid(p)])
    return images_tr, labels_tr


def summarize_volume(path: Path, compute_intensity: bool = False) -> Dict:
    """
    Summarize a single NIfTI volume.

    Only stores the filename (path.name), not the full path -- the full
    path leaks local filesystem info (e.g. C:\\Users\\<name>\\...) into a
    report that may end up checked into the repo, which is unnecessary
    and not something you want on a public GitHub project.

    compute_intensity=True additionally loads pixel data (subsampled) to
    report intensity min/max/mean -- more expensive, used for images only.
    """
    img = nib.load(str(path))
    header = img.header
    shape = header.get_data_shape()
    spacing = header.get_zooms()

    result = {
        "filename": path.name,
        "shape": list(shape),
        "spacing": [round(float(s), 4) for s in spacing],
    }

    if compute_intensity:
        # Subsample every 4th slice to keep this cheap on large volumes.
        dataobj = img.dataobj
        sample = np.asarray(dataobj[:, :, ::4], dtype=np.float32)
        result["intensity_min"] = float(sample.min())
        result["intensity_max"] = float(sample.max())
        result["intensity_mean"] = round(float(sample.mean()), 2)

    return result


def check_dataset() -> Dict:
    images, labels = get_nifti_paths(config.RAW_DIR)
    assert len(images) > 0, f"No images found under {config.RAW_DIR}/imagesTr -- did you download the dataset?"
    assert len(images) == len(labels), (
        f"Number of images ({len(images)}) and labels ({len(labels)}) must match."
    )

    image_stats = [summarize_volume(p, compute_intensity=True) for p in images]
    label_stats = [summarize_volume(p, compute_intensity=False) for p in labels]

    spacings = np.array([s["spacing"] for s in image_stats])
    shapes = np.array([s["shape"] for s in image_stats])

    summary = {
        "n_cases": len(images),
        "spacing_mean": [round(float(x), 4) for x in spacings.mean(axis=0)],
        "spacing_min": [round(float(x), 4) for x in spacings.min(axis=0)],
        "spacing_max": [round(float(x), 4) for x in spacings.max(axis=0)],
        "shape_min": [int(x) for x in shapes.min(axis=0)],
        "shape_max": [int(x) for x in shapes.max(axis=0)],
        "intensity_min_overall": min(s["intensity_min"] for s in image_stats),
        "intensity_max_overall": max(s["intensity_max"] for s in image_stats),
        "configured_target_spacing": list(config.TARGET_SPACING),
        "configured_intensity_clip": list(config.INTENSITY_CLIP),
    }

    stats = {"summary": summary, "images": image_stats, "labels": label_stats}

    out_dir = config.REPORTS_DIR / "data_quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "dataset_stats.json"

    # Write atomically (write to temp file, then rename) so a crash or
    # interrupted run never leaves a truncated/invalid JSON file behind --
    # this is exactly the failure mode the original version of this script hit.
    tmp_path = out_json.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(stats, indent=2))
    tmp_path.replace(out_json)

    print(f"Wrote dataset stats for {len(images)} cases to {out_json}")
    print(f"Spacing range: {summary['spacing_min']} to {summary['spacing_max']}")
    print(f"Intensity range: [{summary['intensity_min_overall']:.1f}, {summary['intensity_max_overall']:.1f}]")
    print(
        f"Configured INTENSITY_CLIP={config.INTENSITY_CLIP} -- "
        f"check this against the observed range above to confirm it's a sane clip window."
    )

    return stats


if __name__ == "__main__":
    check_dataset()
