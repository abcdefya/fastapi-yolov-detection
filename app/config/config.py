from pathlib import Path
import os

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data directories
MODEL_PATH = BASE_DIR / "data" / "models" / "yolov8n.pt"
TEMP_DIR = BASE_DIR / "data" / "temp"

# Create temp directory
TEMP_DIR.mkdir(parents=True, exist_ok=True)