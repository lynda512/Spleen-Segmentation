import json
from pathlib import Path
from typing import Dict, List, Tuple

from monai.data import Dataset, CacheDataset, DataLoader, load_decathlon_datalist
from src import config
from src.data.transforms import get_train_transforms, get_val_transforms


def get_datalist_spleen(
    root: Path,
    split_ratio: float = config.TRAIN_SPLIT,
    seed: int = config.SEED
 ) -> Tuple[List[Dict], List[Dict]]:
    """
    Expect MSD Task09_Spleen structure:
      imagesTr/, labelsTr/, dataset.json
    """
    dataset_json = root / "dataset.json"
    if not dataset_json.exists():
        raise FileNotFoundError(f"dataset.json not found at {dataset_json}")

    full_list = load_decathlon_datalist(
        str(dataset_json),
        is_training=True,
        data_list_key="training",
        base_dir=str(root),
    )
    # Simple split by index; could be improved
    n_train = int(len(full_list) * split_ratio)
    train_files = full_list[:n_train]
    val_files = full_list[n_train:]
    return train_files, val_files


def get_dataloaders() -> Tuple[DataLoader, DataLoader]:
    train_files, val_files = get_datalist_spleen(config.RAW_DIR)

    train_ds = CacheDataset(
        data=train_files,
        transform=get_train_transforms(),
        cache_rate=1.0,
        num_workers=config.NUM_WORKERS,
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=get_val_transforms(),
        cache_rate=1.0,
        num_workers=config.NUM_WORKERS,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    return train_loader, val_loader
