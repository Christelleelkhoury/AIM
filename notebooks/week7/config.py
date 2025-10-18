# config.py
from pathlib import Path
import os

# Root directory (the folder where this file lives)
ROOT = Path(__file__).parent

# ---- Back-compat shim ----
# Some older modules may still import/use LOADER_DIR.
# Keep this so imports like "from config import LOADER_DIR" don't crash.
LOADER_DIR = os.fspath(ROOT)

# Where your trained model checkpoint is stored
CHECKPOINT_PATH = ROOT / "resources"

# Number of output classes (binary: Normal vs Pneumonia)
NUM_CLASSES = 2

# Label mapping for visualization
LABEL_MAP = {
    0: "Normal",
    1: "Pneumonia",
}

# Image preprocessing size (height, width)
IMAGE_SIZE = (1024, 1024)
