"""
Sample Data Preparation Utility
Generates sample leaf images for interactive testing in static/images/samples/
and populates a mini dataset for quick-dev verification of train.py and evaluate.py.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import config

SAMPLES = [
    {
        "filename": "Tomato_Early_Blight.jpg",
        "class_name": "Tomato___Early_blight",
        "bg_color": (34, 139, 34),
        "leaf_color": (46, 125, 50),
        "lesion_type": "concentric",
        "lesion_color": (101, 67, 33),
    },
    {
        "filename": "Tomato_Healthy.jpg",
        "class_name": "Tomato___healthy",
        "bg_color": (34, 139, 34),
        "leaf_color": (56, 142, 60),
        "lesion_type": "none",
        "lesion_color": None,
    },
    {
        "filename": "Potato_Early_Blight.jpg",
        "class_name": "Potato___Early_blight",
        "bg_color": (46, 125, 50),
        "leaf_color": (43, 108, 45),
        "lesion_type": "concentric",
        "lesion_color": (80, 50, 20),
    },
    {
        "filename": "Corn_Common_Rust.jpg",
        "class_name": "Corn_(maize)___Common_rust_",
        "bg_color": (76, 140, 43),
        "leaf_color": (85, 139, 47),
        "lesion_type": "pustules",
        "lesion_color": (180, 82, 42),
    },
    {
        "filename": "Apple_Scab.jpg",
        "class_name": "Apple___Apple_scab",
        "bg_color": (46, 117, 89),
        "leaf_color": (38, 102, 77),
        "lesion_type": "velvety_spots",
        "lesion_color": (50, 40, 20),
    },
]


def draw_leaf_image(spec: dict, size=(224, 224), seed=42) -> Image.Image:
    """Draws a synthetic leaf specimen with appropriate pathological lesions."""
    np.random.seed(seed)
    w, h = size
    # Background neutral canvas (similar to PlantVillage lab setup)
    img = Image.new("RGB", size, (235, 238, 235))
    draw = ImageDraw.Draw(img)

    # Leaf shape (ellipse or polygon)
    leaf_bbox = [int(w * 0.15), int(h * 0.10), int(w * 0.85), int(h * 0.90)]
    draw.ellipse(leaf_bbox, fill=spec["leaf_color"], outline=(25, 80, 30), width=2)

    # Midrib vein
    draw.line([(w // 2, int(h * 0.12)), (w // 2, int(h * 0.88))], fill=(80, 160, 70), width=3)

    # Lateral veins
    for y in range(int(h * 0.25), int(h * 0.80), 25):
        draw.line([(w // 2, y), (int(w * 0.22), y - 15)], fill=(80, 160, 70), width=1)
        draw.line([(w // 2, y), (int(w * 0.78), y - 15)], fill=(80, 160, 70), width=1)

    # Draw pathological symptoms
    ltype = spec["lesion_type"]
    if ltype == "concentric":
        # Alternaria concentric target rings
        centers = [(w // 2 - 25, h // 2 - 20), (w // 2 + 20, h // 2 + 30), (w // 2 - 10, h // 2 + 50)]
        for cx, cy in centers:
            # Yellow chlorotic halo
            draw.ellipse([cx - 24, cy - 24, cx + 24, cy + 24], fill=(218, 165, 32, 100))
            # Outer dark ring
            draw.ellipse([cx - 18, cy - 18, cx + 18, cy + 18], fill=spec["lesion_color"])
            # Mid ring
            draw.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], fill=(120, 80, 40))
            # Center necrotic spot
            draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(40, 25, 10))
    elif ltype == "pustules":
        # Puccinia rust pustules
        for _ in range(30):
            rx = int(np.random.uniform(w * 0.3, w * 0.7))
            ry = int(np.random.uniform(h * 0.2, h * 0.8))
            draw.ellipse([rx - 4, ry - 6, rx + 4, ry + 6], fill=spec["lesion_color"])
    elif ltype == "velvety_spots":
        # Apple scab olive/brown lesions
        for _ in range(12):
            rx = int(np.random.uniform(w * 0.25, w * 0.75))
            ry = int(np.random.uniform(h * 0.2, h * 0.8))
            rad = int(np.random.uniform(8, 16))
            draw.ellipse([rx - rad, ry - rad, rx + rad, ry + rad], fill=spec["lesion_color"])

    # Subtle gaussian blur to smooth edges
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    return img


def setup_sample_assets():
    """Generates sample leaf images into static/images/samples/."""
    config.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Generating sample leaf images in: {config.SAMPLES_DIR}")
    for idx, spec in enumerate(SAMPLES):
        img = draw_leaf_image(spec, seed=42 + idx)
        out_path = config.SAMPLES_DIR / spec["filename"]
        img.save(out_path, quality=95)
        print(f"  - Saved sample: {spec['filename']}")


def setup_quick_dev_dataset():
    """Populates dataset/ with mini sample classes for quick pipeline verification."""
    print(f"\n[INFO] Setting up mini dataset in: {config.DATASET_DIR}")
    for idx, spec in enumerate(SAMPLES):
        class_dir = config.DATASET_DIR / spec["class_name"]
        class_dir.mkdir(parents=True, exist_ok=True)
        # Generate 10 images per class for fast testing
        for i in range(10):
            img = draw_leaf_image(spec, seed=idx * 100 + i)
            img.save(class_dir / f"leaf_{i:03d}.jpg", quality=90)
    print(f"[INFO] Mini dataset populated across {len(SAMPLES)} classes.")


if __name__ == "__main__":
    setup_sample_assets()
    setup_quick_dev_dataset()
    print("\n[SUCCESS] Sample data generation completed successfully!")
