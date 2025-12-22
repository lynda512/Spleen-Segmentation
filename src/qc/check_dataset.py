import nibabel as nib
from pathlib import Path
import json

from src import config  # or `from spleen_qc_seg import config` if you use the package layout


def get_nifti_paths(root: Path):
    def valid(p):
        return p.name.endswith(".nii.gz") and not p.name.startswith("._")

    images_tr = sorted([p for p in (root / "imagesTr").glob("*.nii.gz") if valid(p)])
    labels_tr = sorted([p for p in (root / "labelsTr").glob("*.nii.gz") if valid(p)])
    return images_tr, labels_tr


def summarize_header(path: Path):
    img = nib.load(str(path))
    header = img.header
    shape = header.get_data_shape()
    spacing = header.get_zooms()
    return {
        "path": path.name,
        "shape": shape,
        "spacing": spacing,
    }


def summarize_label_header(path: Path):
    img = nib.load(str(path))
    header = img.header
    shape = header.get_data_shape()
    spacing = header.get_zooms()
    return {
        "path": path.name,
        "shape": shape,
        "spacing": spacing,
    }


def check_dataset():
    images, labels = get_nifti_paths(config.RAW_DIR)
    assert len(images) == len(labels), "Number of images and labels must match."

    stats = {"images": [], "labels": []}

    for img_path in images:
        stats["images"].append(summarize_header(img_path))

    for lbl_path in labels:
        stats["labels"].append(summarize_header(lbl_path))

    out_dir = config.REPORTS_DIR / "data_quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "dataset_stats.json"
    with out_json.open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote basic dataset stats to {out_json}")

if __name__ == "__main__":
    check_dataset()
