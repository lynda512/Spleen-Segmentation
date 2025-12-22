from pathlib import Path

# Base paths
# NOTE: parents[1] points to the project root (parents[2] went one level too high)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "Task09_Spleen"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Training config
NUM_EPOCHS = 5
BATCH_SIZE = 16
LR = 1e-4
VAL_INTERVAL = 1
SEED = 42

# Data config
TARGET_SPACING = (1.5, 1.5, 1.5)  # mm
INTENSITY_CLIP = (-200, 200)
NUM_WORKERS = 4
TRAIN_SPLIT = 0.8

# Model config
IN_CHANNELS = 1
OUT_CHANNELS = 2  # background + spleen
