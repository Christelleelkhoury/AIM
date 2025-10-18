# config.py
from pathlib import Path

# Root directory (the folder where this file lives)
ROOT = Path(__file__).parent

# Where your trained model checkpoint is stored
CHECKPOINT_PATH = ROOT / "resources"

# Number of output classes (for binary pneumonia vs normal)
NUM_CLASSES = 2

# Label mapping for visualization
LABEL_MAP = {
    0: "Normal",
    1: "Pneumonia"
}

# Image preprocessing size (height, width)
IMAGE_SIZE = (1024, 1024)
