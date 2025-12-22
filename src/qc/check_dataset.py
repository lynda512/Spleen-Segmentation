from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np
import json

from .. import config


def get_nifti_paths(root: Path):
    images_tr = sorted((root / "imagesTr").glob("*.nii.gz"))
    labels_tr = sorted((root / "labelsTr").glob("*.nii.gz"))
    return images_tr, labels_tr


def summarize_volume(path: Path) -> Dict:
    img = nib.load(str(path))
    data = img.get_fdata()
    return {
        "path": str(path),
        "shape": data.shape,
        "spacing": img.header.get_zooms(),
        "nan_fraction": float(np.isnan(data).mean()),
        "min": float(np.nanmin(data)),
        "max": float(np.nanmax(data)),
    }


def check_dataset():
    images, labels = get_nifti_paths(config.RAW_DIR)
    assert len(images) == len(labels), "Number of images and labels must match."

    stats = {"images": [], "labels": []}

    for img_path in images:
        stats["images"].append(summarize_volume(img_path))

    for lbl_path in labels:
        img = nib.load(str(lbl_path))
        data = img.get_fdata()
        unique_labels = np.unique(data)
        stats["labels"].append(
            {
                "path": str(lbl_path),
                "shape": data.shape,
                "spacing": img.header.get_zooms(),
                "nan_fraction": float(np.isnan(data).mean()),
                "unique_labels": [int(x) for x in unique_labels],
            }
        )

    out_dir = config.REPORTS_DIR / "data_quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "dataset_stats.json"
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote basic dataset stats to {out_json}")


if __name__ == "__main__":
    check_dataset()
