import os
from pathlib import Path

# Base paths
# NOTE: parents[1] points to the project root (parents[2] went one level too high)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# DATA_DIR can be overridden via the SPLEEN_DATA_DIR env var so the same code
# works locally, on Colab (e.g. /content/data), and on Kaggle (e.g. /kaggle/working/data)
# without editing this file per-environment.
DATA_DIR = Path(os.environ.get("SPLEEN_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw" / "Task09_Spleen"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# REPORTS_DIR / CHECKPOINT_DIR similarly overridable so Colab runs can write
# to a Drive-mounted path that survives session resets.
REPORTS_DIR = Path(os.environ.get("SPLEEN_REPORTS_DIR", PROJECT_ROOT / "reports"))
CHECKPOINT_DIR = Path(os.environ.get("SPLEEN_CKPT_DIR", PROJECT_ROOT / "checkpoints"))

# Training config
NUM_EPOCHS = 100
BATCH_SIZE = 4          # patch-based training (96^3 patches), not full volumes
VAL_INTERVAL = 2
LR = 1e-4
WEIGHT_DECAY = 1e-5
SEED = 42
EARLY_STOPPING_PATIENCE = 15   # stop if val Dice hasn't improved in N val checks

# Data config
TARGET_SPACING = (1.5, 1.5, 1.5)  # mm
INTENSITY_CLIP = (-200, 200)
NUM_WORKERS = 4
TRAIN_SPLIT = 0.8
PATCH_SIZE = (96, 96, 96)
NUM_SAMPLES_PER_VOLUME = 4   # patches drawn per volume per epoch (RandCropByPosNegLabeld)

# Model config
IN_CHANNELS = 1
OUT_CHANNELS = 2  # background + spleen

# Experiment tracking
WANDB_PROJECT = "spleen-segmentation"
WANDB_ENTITY = None  # set to your W&B username/team, or leave None to use default
