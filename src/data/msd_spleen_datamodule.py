import random
from pathlib import Path
from typing import Dict, List, Tuple

from monai.data import CacheDataset, DataLoader, load_decathlon_datalist
from src import config
from src.data.transforms import get_train_transforms, get_val_transforms, get_test_transforms


def get_datalist_spleen(
    root: Path,
    split_ratio: float = config.TRAIN_SPLIT,
    seed: int = config.SEED,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Expect MSD Task09_Spleen structure: imagesTr/, labelsTr/, dataset.json

    Splits the labelled training set (41 cases) into train/val/test.
    We use a 70/15/15 split off the original `split_ratio` as the train
    fraction, with the remainder split evenly between val and test --
    this gives a held-out test set for final, unbiased reporting that
    was missing from the original train/val-only split.

    The split is shuffled with a fixed seed so it's reproducible across
    runs (the original implementation used the on-disk file order, which
    is not guaranteed to be random and risks correlated train/val splits
    e.g. if cases were added to the dataset in a non-random order).
    """
    dataset_json = root / "dataset.json"
    if not dataset_json.exists():
        raise FileNotFoundError(
            f"dataset.json not found at {dataset_json}. "
            f"Did you download and extract Task09_Spleen into {root}?"
        )

    full_list = load_decathlon_datalist(
        str(dataset_json),
        is_segmentation=True,
        data_list_key="training",
        base_dir=str(root),
    )

    rng = random.Random(seed)
    shuffled = full_list.copy()
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * split_ratio)
    n_remaining = n_total - n_train
    n_val = n_remaining // 2
    # test gets whatever's left, so we don't silently drop a case to rounding
    n_test = n_remaining - n_val

    train_files = shuffled[:n_train]
    val_files = shuffled[n_train:n_train + n_val]
    test_files = shuffled[n_train + n_val:]

    assert len(train_files) + len(val_files) + len(test_files) == n_total
    return train_files, val_files, test_files


def get_dataloaders(cache_rate: float = 1.0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).

    cache_rate controls MONAI's CacheDataset in-memory caching. On a
    Colab/Kaggle T4 (~12-16GB RAM), cache_rate=1.0 for this dataset
    (41 CT volumes) is generally fine, but drop it if you hit OOM.
    """
    train_files, val_files, test_files = get_datalist_spleen(config.RAW_DIR)

    train_ds = CacheDataset(
        data=train_files,
        transform=get_train_transforms(),
        cache_rate=cache_rate,
        num_workers=config.NUM_WORKERS,
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=get_val_transforms(),
        cache_rate=cache_rate,
        num_workers=config.NUM_WORKERS,
    )
    test_ds = CacheDataset(
        data=test_files,
        transform=get_test_transforms(),
        cache_rate=cache_rate,
        num_workers=config.NUM_WORKERS,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    # val/test use batch_size=1: full preprocessed volumes vary in size after
    # CropForegroundd, so they aren't guaranteed to stack into a batch tensor.
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS)

    return train_loader, val_loader, test_loader
