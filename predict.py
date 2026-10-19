"""
Plant Disease Detection Prediction Module
Handles image loading, RGB conversion, 224x224 resizing, MobileNetV2 preprocessing,
model inference, top-3 candidates calculation, and agricultural recommendations.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import config


class PlantDiseasePredictor:
    """
    Singleton inference engine for plant leaf disease classification.
    Loads model weights and class mappings once into memory.
    """

    _instance: Optional["PlantDiseasePredictor"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PlantDiseasePredictor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        model_path: Optional[Path] = None,
        class_names_path: Optional[Path] = None,
        disease_info_path: Optional[Path] = None,
    ):
        if self._initialized:
            return

        self.model_path = Path(model_path or config.MODEL_PATH)
        self.class_names_path = Path(class_names_path or config.CLASS_NAMES_PATH)
        self.disease_info_path = Path(disease_info_path or config.DISEASE_INFO_PATH)

        self.model = None
        self.class_mapping: Dict[int, str] = {}
        self.disease_info: Dict[str, Dict[str, Any]] = {}
        self.is_model_loaded: bool = False

        self._load_disease_info()
        self._load_model_and_classes()
        self._initialized = True

    def _load_disease_info(self):
        """Loads disease descriptions, symptoms, and recommendations from JSON."""
        if self.disease_info_path.exists():
            try:
                with open(self.disease_info_path, "r", encoding="utf-8") as f:
                    self.disease_info = json.load(f)
                print(f"[INFO] Loaded disease metadata for {len(self.disease_info)} classes.")
            except Exception as e:
                print(f"[WARNING] Could not parse disease metadata JSON: {e}")
                self.disease_info = {}
        else:
            print(f"[WARNING] Disease metadata file not found at: {self.disease_info_path}")
            self.disease_info = {}

    def _load_model_and_classes(self):
        """Loads the trained Keras model and exact class index mapping."""
        if not self.model_path.exists() or not self.class_names_path.exists():
            print(f"[WARNING] Trained model '{self.model_path}' or class mapping not found.")
            print("[INFO] Inference engine will run in fallback mock mode until model is trained.")
            self.is_model_loaded = False
            return

        try:
            import tensorflow as tf

            print(f"[INFO] Loading trained Keras model from {self.model_path}...")
            self.model = tf.keras.models.load_model(self.model_path)
            
            with open(self.class_names_path, "r", encoding="utf-8") as f:
                raw_mapping = json.load(f)
            # Normalize keys to integers
            self.class_mapping = {int(k): v for k, v in raw_mapping.items()}
            self.is_model_loaded = True
            print(f"[INFO] Successfully loaded model with {len(self.class_mapping)} classes.")
        except Exception as e:
            print(f"[ERROR] Failed to load trained model: {e}")
            self.is_model_loaded = False

    def preprocess_image(self, image_input: Any) -> np.ndarray:
        """
        Safely validates, converts to RGB, and resizes image to (224, 224).
        Accepts: file path (str/Path), file-like object, or PIL.Image.
        Returns: NumPy array of shape (1, 224, 224, 3) in float32 format.
        """
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            pil_img = Image.open(path)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
        else:
            # File-like stream (e.g. Flask request.files['image'])
            pil_img = Image.open(image_input)

        # Convert to RGB (handles RGBA, Grayscale, CMYK, etc.)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        # Resize to standard MobileNetV2 input dimensions
        pil_img = pil_img.resize(config.IMAGE_SIZE, Image.Resampling.BILINEAR)

        # Convert to numpy array
        img_array = np.array(pil_img, dtype=np.float32)

        # Add batch dimension: (1, 224, 224, 3)
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch

    def _parse_raw_class_name(self, raw_name: str) -> Tuple[str, str]:
        """Parses raw class names like 'Tomato___Early_blight' into clean parts."""
        if "___" in raw_name:
            parts = raw_name.split("___")
            crop = parts[0].replace("_", " ").strip()
            condition = parts[1].replace("_", " ").strip()
        else:
            parts = raw_name.split("_")
            crop = parts[0] if parts else "Plant"
            condition = " ".join(parts[1:]) if len(parts) > 1 else "Condition"
        return crop, condition

    def predict(self, image_input: Any) -> Dict[str, Any]:
        """
        Executes complete prediction pipeline on an input leaf image:
        1. Preprocessing (safe open, RGB convert, 224x224 resize)
        2. Model inference
        3. Class probabilities extraction
        4. Top-3 candidates calculation
        5. Disease metadata and agricultural recommendations retrieval
        """
        img_batch = self.preprocess_image(image_input)

        if not self.is_model_loaded:
            # Fallback demonstrator when model is not yet trained
            return self._fallback_prediction()

        # Run inference
        probabilities = self.model.predict(img_batch, verbose=0)[0]

        # Top prediction
        predicted_idx = int(np.argmax(probabilities))
        raw_class = self.class_mapping.get(predicted_idx, "Unknown___Unknown")
        confidence = float(probabilities[predicted_idx])
        confidence_pct = round(confidence * 100, 2)

        # Top-3 predictions
        top_indices = np.argsort(probabilities)[::-1][: config.TOP_K_PREDICTIONS]
        top_3 = []
        for idx in top_indices:
            cls_name = self.class_mapping.get(int(idx), "Unknown")
            c_crop, c_cond = self._parse_raw_class_name(cls_name)
            top_3.append(
                {
                    "class_key": cls_name,
                    "crop": c_crop,
                    "condition": c_cond,
                    "confidence": round(float(probabilities[idx]) * 100, 2),
                }
            )

        # Retrieve disease details
        details = self.disease_info.get(raw_class)
        if details:
            crop_name = details.get("crop_name", "Unknown Crop")
            disease_name = details.get("disease_name", "Unknown Condition")
            description = details.get("description", "No description available.")
            symptoms = details.get("symptoms", [])
            recommendations = details.get("recommendations", {})
            severity = details.get("severity", "Moderate")
            is_healthy = details.get("is_healthy", False)
        else:
            crop_name, disease_name = self._parse_raw_class_name(raw_class)
            is_healthy = "healthy" in disease_name.lower()
            description = f"Detected condition: {disease_name} affecting {crop_name} foliage."
            symptoms = [f"Observable leaf stress associated with {disease_name}"]
            recommendations = {
                "cultural": "Ensure adequate spacing, sanitize tools, and inspect neighboring plants.",
                "organic": "Apply biological or organic protective sprays if symptoms persist.",
                "chemical": "Consult local agricultural extension service for registered products.",
                "prevention": "Rotate crops and avoid overhead irrigation to minimize leaf wetness.",
            }
            severity = "None" if is_healthy else "Moderate"

        # Determine confidence level & caution warnings
        if confidence >= 0.80:
            confidence_level = "High"
            confidence_badge = "success"
            warning_message = None
        elif confidence >= config.CONFIDENCE_THRESHOLD:
            confidence_level = "Moderate"
            confidence_badge = "warning"
            warning_message = (
                "Moderate prediction confidence. The visual symptoms may share characteristics "
                "with related disorders. Verify with additional foliar checks."
            )
        else:
            confidence_level = "Low"
            confidence_badge = "danger"
            warning_message = (
                "Low prediction confidence. The uploaded image might have low contrast, atypical angles, "
                "or may belong to an unsupported plant variety. Laboratory or agronomic confirmation is advised."
            )

        return {
            "success": True,
            "raw_class": raw_class,
            "crop": crop_name,
            "condition": disease_name,
            "confidence": confidence_pct,
            "confidence_ratio": confidence,
            "confidence_level": confidence_level,
            "confidence_badge": confidence_badge,
            "is_healthy": is_healthy,
            "severity": severity,
            "description": description,
            "symptoms": symptoms,
            "recommendations": recommendations,
            "top_3": top_3,
            "warning": warning_message,
            "disclaimer": (
                "Model confidence represents estimated class probabilities based on the PlantVillage dataset "
                "and does not constitute a guaranteed clinical diagnostic result."
            ),
        }

    def _fallback_prediction(self) -> Dict[str, Any]:
        """Provides a simulated response when weights have not been trained yet."""
        return {
            "success": False,
            "crop": "Model Not Yet Trained",
            "condition": "Training Required",
            "confidence": 0.0,
            "confidence_ratio": 0.0,
            "confidence_level": "None",
            "confidence_badge": "secondary",
            "is_healthy": False,
            "severity": "N/A",
            "description": (
                "The MobileNetV2 model weights ('plant_disease_model.keras') are not found. "
                "Please run 'python train.py' to train the model, or check setup instructions in README.md."
            ),
            "symptoms": ["Model weights missing from models/ directory."],
            "recommendations": {
                "cultural": "Open a terminal in the project directory.",
                "organic": "Execute: python train.py --quick-dev (for quick test) or python train.py (for full training).",
                "chemical": "Restart the Flask app once training completes.",
                "prevention": "Ensure dataset is placed in dataset/ before running full training.",
            },
            "top_3": [],
            "warning": "Model weights missing. Please train the model to enable live predictions.",
            "disclaimer": "System in setup mode.",
        }


def get_predictor() -> PlantDiseasePredictor:
    """Convenience getter for the singleton predictor."""
    return PlantDiseasePredictor()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Plant Disease Predictor on single image")
    parser.add_argument("--image", type=str, required=True, help="Path to leaf image file")
    args = parser.parse_args()

    predictor = get_predictor()
    try:
        result = predictor.predict(args.image)
        print("\n" + "=" * 60)
        print("PREDICTION RESULT:")
        print("=" * 60)
        print(f"Crop:            {result['crop']}")
        print(f"Condition:       {result['condition']}")
        print(f"Confidence:      {result['confidence']}% ({result['confidence_level']})")
        print(f"Severity:        {result['severity']}")
        print(f"Description:     {result['description']}")
        print("\nRecommendations:")
        for category, advice in result["recommendations"].items():
            print(f"  [{category.capitalize()}]: {advice}")
        print("=" * 60)
    except Exception as err:
        print(f"[ERROR] Inference failed: {err}")
        sys.exit(1)
