"""
Plant Disease Detection Model Training Module
Uses MobileNetV2 with Transfer Learning and Data Augmentation.
Saves trained weights in modern .keras format and records class mapping in JSON.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np

# Import configuration constants
import config


def parse_arguments():
    """Parse command line arguments for training options."""
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 Transfer Learning Model on PlantVillage Dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(config.DATASET_DIR),
        help="Path to dataset root folder containing class subdirectories",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help="Number of epochs for initial transfer learning training",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help="Batch size for training and validation",
    )
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Enable optional fine-tuning phase by unfreezing upper MobileNetV2 layers",
    )
    parser.add_argument(
        "--quick-dev",
        action="store_true",
        help="Run quick development test mode with reduced samples and 2 epochs",
    )
    return parser.parse_args()


def check_and_prepare_dataset(dataset_path: Path):
    """
    Verify dataset existence and directory structure.
    Returns:
        tuple (class_names: list, is_pre_split: bool)
    """
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset directory '{dataset_path}' does not exist. "
            "Please download the PlantVillage dataset from Kaggle and place it in the 'dataset/' folder. "
            "See README.md for instructions or run with --quick-dev after generating sample data."
        )

    # Check if pre-split (train/val/test) or single folder of classes
    subdirs = [d for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith(".")]

    if {"train", "val"}.issubset({d.name.lower() for d in subdirs}):
        # Pre-split structure
        train_dir = dataset_path / "train"
        class_folders = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        return class_folders, True
    else:
        # Direct class subfolders
        class_folders = sorted([d.name for d in subdirs if d.name not in ["results", "models", "uploads"]])
        if not class_folders:
            raise ValueError(
                f"No class subfolders found inside '{dataset_path}'. "
                "Ensure each disease class has its own folder containing image files."
            )
        return class_folders, False


def build_transfer_model(num_classes: int, input_shape=config.IMAGE_SHAPE):
    """
    Builds the MobileNetV2 transfer learning architecture:
    1. Pretrained MobileNetV2 base (ImageNet weights, top removed)
    2. Frozen base weights
    3. Global Average Pooling 2D
    4. Batch Normalization & Dropout for regularization
    5. Dense ReLU feature layer
    6. Dense Softmax output layer
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2

    print("\n[INFO] Loading MobileNetV2 base pretrained on ImageNet...")
    try:
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
    except Exception as exc:
        print(f"\n[ERROR] Failed to load ImageNet pretrained weights: {exc}")
        print("Please check your internet connection or proxy settings.")
        raise exc

    # Freeze base model parameters initially
    base_model.trainable = False

    # Construct custom classification head
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    
    # Preprocessing layer: MobileNetV2 expects pixel values in [-1, 1]
    # tf.keras.applications.mobilenet_v2.preprocess_input scales [0, 255] to [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.BatchNormalization(name="batch_norm_1")(x)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="dense_features")(x)
    x = layers.BatchNormalization(name="batch_norm_2")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="PlantDiseaseMobileNetV2")

    # Compile with Adam and Categorical Cross-Entropy
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.INITIAL_LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=min(3, num_classes), name="top_3_accuracy"),
        ],
    )

    print("\n[INFO] Model Architecture Summary:")
    model.summary(line_length=90)
    return model, base_model


def plot_and_save_history(history, output_path: Path):
    """Save training and validation loss/accuracy curves to disk."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs_range = range(1, len(history.history["loss"]) + 1)

    # Plot Accuracy
    ax1.plot(epochs_range, history.history["accuracy"], "g-o", label="Training Accuracy", linewidth=2)
    if "val_accuracy" in history.history:
        ax1.plot(epochs_range, history.history["val_accuracy"], "b--s", label="Validation Accuracy", linewidth=2)
    ax1.set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.7)

    # Plot Loss
    ax2.plot(epochs_range, history.history["loss"], "r-o", label="Training Loss", linewidth=2)
    if "val_loss" in history.history:
        ax2.plot(epochs_range, history.history["val_loss"], "m--s", label="Validation Loss", linewidth=2)
    ax2.set_title("Training & Validation Loss", fontsize=14, fontweight="bold", pad=12)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Categorical Cross-Entropy Loss", fontsize=12)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved training progress plot to: {output_path}")


def train():
    """Main training routine."""
    args = parse_arguments()
    dataset_path = Path(args.dataset)

    print("=" * 80)
    print("PLANT DISEASE DETECTION: MOBILENETV2 TRANSFER LEARNING TRAINING")
    print("=" * 80)

    try:
        import tensorflow as tf
    except ImportError:
        print("[ERROR] TensorFlow is not installed. Run: pip install tensorflow")
        sys.exit(1)

    # Reproducibility seed
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)

    # Check dataset structure
    class_names, is_pre_split = check_and_prepare_dataset(dataset_path)
    num_classes = len(class_names)
    print(f"[INFO] Detected {num_classes} plant disease classes:")
    for idx, cname in enumerate(class_names):
        print(f"  [{idx:02d}] {cname}")

    # Build Class Index Mapping and save immediately
    class_mapping = {idx: cname for idx, cname in enumerate(class_names)}
    with open(config.CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2)
    print(f"\n[INFO] Saved class names index mapping to: {config.CLASS_NAMES_PATH}")

    # Data Augmentation pipeline
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomZoom(0.15),
        ],
        name="data_augmentation",
    )

    batch_size = args.batch_size if not args.quick_dev else 8
    epochs = args.epochs if not args.quick_dev else 2

    # Load datasets using image_dataset_from_directory
    print("\n[INFO] Loading image datasets...")
    if is_pre_split:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_path / "train",
            image_size=config.IMAGE_SIZE,
            batch_size=batch_size,
            label_mode="categorical",
            shuffle=True,
            seed=config.RANDOM_SEED,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_path / "val",
            image_size=config.IMAGE_SIZE,
            batch_size=batch_size,
            label_mode="categorical",
            shuffle=False,
        )
    else:
        # Split single directory into 70% train, 15% val, 15% test
        raw_train_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_path,
            validation_split=config.VAL_RATIO + config.TEST_RATIO,
            subset="training",
            seed=config.RANDOM_SEED,
            image_size=config.IMAGE_SIZE,
            batch_size=batch_size,
            label_mode="categorical",
        )
        val_test_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_path,
            validation_split=config.VAL_RATIO + config.TEST_RATIO,
            subset="validation",
            seed=config.RANDOM_SEED,
            image_size=config.IMAGE_SIZE,
            batch_size=batch_size,
            label_mode="categorical",
        )

        # Further partition val_test into 50% val and 50% test (each ~15% of total)
        val_test_batches = tf.data.experimental.cardinality(val_test_ds).numpy()
        val_batches = max(1, val_test_batches // 2)
        val_ds = val_test_ds.take(val_batches)
        test_ds = val_test_ds.skip(val_batches)
        train_ds = raw_train_ds

    # Apply augmentation to training dataset
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Performance optimization (prefetch & cache)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Build the MobileNetV2 Transfer Learning Model
    model, base_model = build_transfer_model(num_classes)

    # Training Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print(f"\n[INFO] Starting Phase 1 Training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Optional Fine-Tuning Stage
    if args.fine_tune and not args.quick_dev:
        print("\n" + "=" * 80)
        print("[INFO] Starting Phase 2: Fine-Tuning Upper Layers of MobileNetV2...")
        print("=" * 80)
        base_model.trainable = True
        # Freeze all layers before the 100th layer
        fine_tune_at = 100
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.FINE_TUNE_LEARNING_RATE),
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.TopKCategoricalAccuracy(k=min(3, num_classes), name="top_3_accuracy"),
            ],
        )

        fine_tune_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(config.MODEL_PATH),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
        ]

        history_fine = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.FINE_TUNE_EPOCHS,
            callbacks=fine_tune_callbacks,
            verbose=1,
        )

    # Explicitly save final model
    print(f"\n[INFO] Saving final model to: {config.MODEL_PATH}")
    model.save(config.MODEL_PATH)

    # Plot and save learning curves
    plot_and_save_history(history, config.TRAINING_PLOT_PATH)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE SUCCESSFULLY!")
    print(f"Model saved to:       {config.MODEL_PATH}")
    print(f"Class names saved to: {config.CLASS_NAMES_PATH}")
    print(f"Plot saved to:        {config.TRAINING_PLOT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    train()
