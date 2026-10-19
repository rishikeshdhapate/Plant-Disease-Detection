"""
Project Configuration Module
Centralizes all paths, hyperparameters, and application settings for
the Plant Disease Detection Deep Learning system.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
UPLOADS_DIR = (
    Path(os.environ.get("VERCEL_TMP_DIR", "/tmp/plant-disease-uploads"))
    if os.environ.get("VERCEL") == "1"
    else BASE_DIR / "uploads"
)
DISEASE_INFO_DIR = BASE_DIR / "disease_info"
STATIC_DIR = BASE_DIR / "static"
SAMPLES_DIR = STATIC_DIR / "images" / "samples"

# Model & Artifact File Paths
MODEL_PATH = MODELS_DIR / "plant_disease_model.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
DISEASE_INFO_PATH = DISEASE_INFO_DIR / "disease_details.json"
TRAINING_PLOT_PATH = RESULTS_DIR / "training_history.png"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
EVALUATION_REPORT_PATH = RESULTS_DIR / "evaluation_metrics.json"

# Image & Data Preprocessing
IMAGE_SIZE = (224, 224)
IMAGE_SHAPE = (224, 224, 3)
CHANNELS = 3

# Deep Learning Hyperparameters
BATCH_SIZE = 32
EPOCHS = 15
INITIAL_LEARNING_RATE = 1e-3
FINE_TUNE_EPOCHS = 5
FINE_TUNE_LEARNING_RATE = 1e-5

# Dataset Partitioning (Stratified)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Web Application & Inference Settings
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
CONFIDENCE_THRESHOLD = 0.60  # Flag predictions below 60% with warning
TOP_K_PREDICTIONS = 3

# Vercel's deployed bundle is read-only; only its temporary upload directory is writable.
if os.environ.get("VERCEL") == "1":
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
else:
    for directory in [
        DATASET_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        UPLOADS_DIR,
        DISEASE_INFO_DIR,
        SAMPLES_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    """Check if the provided filename has an approved image extension."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS
