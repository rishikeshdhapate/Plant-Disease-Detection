"""
Plant Disease Detection Model Evaluation Module
Evaluates trained MobileNetV2 model on held-out test data.
Generates Confusion Matrix heatmap, Classification Report, Precision, Recall,
F1-score, and Overall Test Accuracy.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import config


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate Plant Disease Detection Model")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(config.DATASET_DIR),
        help="Path to dataset root folder",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(config.MODEL_PATH),
        help="Path to trained .keras model file",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default=str(config.CLASS_NAMES_PATH),
        help="Path to class names JSON mapping",
    )
    return parser.parse_args()


def load_model_and_classes(model_path: Path, class_names_path: Path):
    """Load model weights and class index mapping with clear error messages."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please train the model first by running: python train.py"
        )
    if not class_names_path.exists():
        raise FileNotFoundError(
            f"Class names mapping not found at '{class_names_path}'. "
            "Please train the model first or verify training outputs."
        )

    import tensorflow as tf

    print(f"[INFO] Loading trained model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    with open(class_names_path, "r", encoding="utf-8") as f:
        class_mapping = json.load(f)

    # Convert keys to int if necessary
    class_names = [class_mapping[str(i)] if str(i) in class_mapping else class_mapping[i] for i in range(len(class_mapping))]
    return model, class_names


def evaluate():
    args = parse_arguments()
    model_path = Path(args.model)
    class_names_path = Path(args.class_names)
    dataset_path = Path(args.dataset)

    print("=" * 80)
    print("PLANT DISEASE DETECTION: MODEL EVALUATION & METRICS")
    print("=" * 80)

    try:
        import tensorflow as tf
    except ImportError:
        print("[ERROR] TensorFlow is required. Run: pip install tensorflow")
        sys.exit(1)

    # Load model and class names
    model, class_names = load_model_and_classes(model_path, class_names_path)
    num_classes = len(class_names)

    # Determine test dataset
    if (dataset_path / "test").exists():
        test_dir = dataset_path / "test"
        print(f"[INFO] Loading dedicated test set from '{test_dir}'...")
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=config.IMAGE_SIZE,
            batch_size=config.BATCH_SIZE,
            label_mode="categorical",
            shuffle=False,
        )
    elif dataset_path.exists():
        print(f"[INFO] Extracting held-out test split from '{dataset_path}'...")
        val_test_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_path,
            validation_split=config.VAL_RATIO + config.TEST_RATIO,
            subset="validation",
            seed=config.RANDOM_SEED,
            image_size=config.IMAGE_SIZE,
            batch_size=config.BATCH_SIZE,
            label_mode="categorical",
            shuffle=False,
        )
        val_test_batches = int(tf.data.experimental.cardinality(val_test_ds).numpy())
        if val_test_batches > 1:
            val_batches = val_test_batches // 2
            test_ds = val_test_ds.skip(val_batches)
        else:
            test_ds = val_test_ds
    else:
        raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist.")

    # Collect ground truth labels and predictions
    print("\n[INFO] Running inference across test dataset batches...")
    y_true = []
    y_pred_probs = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred_probs.extend(preds)

    total_samples = len(y_true)
    if total_samples == 0:
        print("[WARNING] No test samples found to evaluate.")
        return

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate overall metrics
    overall_accuracy = float(np.mean(y_true == y_pred))
    
    # Calculate top-3 accuracy
    top_3_correct = 0
    for i in range(total_samples):
        top_3_indices = np.argsort(y_pred_probs[i])[-3:]
        if y_true[i] in top_3_indices:
            top_3_correct += 1
    top_3_accuracy = float(top_3_correct / total_samples)

    # Scikit-learn Classification Report
    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        labels=range(num_classes),
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        labels=range(num_classes),
        zero_division=0,
    )
    print(report_text)

    # Confusion Matrix
    print("\n[INFO] Generating Confusion Matrix heatmap...")
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    # Normalize confusion matrix by true class row sums
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.nan_to_num(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis])

    fig, ax = plt.subplots(figsize=(max(10, num_classes * 0.4), max(8, num_classes * 0.35)))
    sns.heatmap(
        cm_norm,
        annot=num_classes <= 15,  # Annotate numbers if classes count is moderate
        fmt=".2f" if num_classes <= 15 else "d",
        cmap="YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=True,
    )
    ax.set_title(
        f"Plant Disease Classification Confusion Matrix\nOverall Test Accuracy: {overall_accuracy * 100:.2f}%",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Predicted Condition", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Condition", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    fig.savefig(config.CONFUSION_MATRIX_PATH, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved confusion matrix heatmap to: {config.CONFUSION_MATRIX_PATH}")

    # Save summary metrics to JSON
    summary_metrics = {
        "total_test_samples": total_samples,
        "overall_accuracy": round(overall_accuracy, 4),
        "overall_accuracy_percentage": f"{overall_accuracy * 100:.2f}%",
        "top_3_accuracy": round(top_3_accuracy, 4),
        "macro_avg_f1_score": round(report_dict.get("macro avg", {}).get("f1-score", 0.0), 4),
        "weighted_avg_f1_score": round(report_dict.get("weighted avg", {}).get("f1-score", 0.0), 4),
        "class_details": {
            cname: {
                "precision": round(report_dict.get(cname, {}).get("precision", 0.0), 4),
                "recall": round(report_dict.get(cname, {}).get("recall", 0.0), 4),
                "f1_score": round(report_dict.get(cname, {}).get("f1-score", 0.0), 4),
                "support": report_dict.get(cname, {}).get("support", 0),
            }
            for cname in class_names
        },
    }

    with open(config.EVALUATION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"[INFO] Saved evaluation metrics JSON to: {config.EVALUATION_REPORT_PATH}")

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print(f"Total Test Samples:  {total_samples}")
    print(f"Overall Accuracy:    {overall_accuracy * 100:.2f}%")
    print(f"Top-3 Accuracy:      {top_3_accuracy * 100:.2f}%")
    print(f"Macro Avg F1-Score:  {summary_metrics['macro_avg_f1_score']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    evaluate()
