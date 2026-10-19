"""
Plant Disease Detection Flask Web Application
Provides a modern agricultural dashboard for leaf disease diagnosis,
confidence score visualizers, and disease management recommendations.
"""

import base64
import hmac
import json
import mimetypes
import os
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from werkzeug.utils import secure_filename
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
    send_from_directory,
    abort,
)

import config

VERCEL_DEPLOYMENT = os.environ.get("VERCEL") == "1"
INFERENCE_API_URL = os.environ.get("INFERENCE_API_URL", "").rstrip("/")
INFERENCE_API_TOKEN = os.environ.get("INFERENCE_API_TOKEN", "")

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "plant-disease-secure-key-2026")
app.config["UPLOAD_FOLDER"] = str(config.UPLOADS_DIR)
app.config["MAX_CONTENT_LENGTH"] = (
    3 * 1024 * 1024 if VERCEL_DEPLOYMENT else config.MAX_CONTENT_LENGTH
)

if VERCEL_DEPLOYMENT:
    predictor = None
else:
    from predict import get_predictor

    predictor = get_predictor()


def predict_image(image_path: Path) -> dict[str, Any]:
    """Run local inference or forward an image to the configured inference service."""
    if predictor is not None:
        return predictor.predict(image_path)

    if not INFERENCE_API_URL:
        raise RuntimeError("INFERENCE_API_URL is not configured for this deployment.")

    boundary = f"----FloraGuard{uuid.uuid4().hex}"
    filename = secure_filename(image_path.name)
    image_data = image_path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8") + image_data + f"\r\n--{boundary}--\r\n".encode("utf-8")
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if INFERENCE_API_TOKEN:
        headers["Authorization"] = f"Bearer {INFERENCE_API_TOKEN}"

    api_request = Request(
        f"{INFERENCE_API_URL}/api/predict",
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(api_request, timeout=50) as response:
            return json.loads(response.read())
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Inference service returned HTTP {error.code}: {detail}"
        ) from error
    except URLError as error:
        raise RuntimeError(f"Could not reach inference service: {error.reason}") from error


def cleanup_old_uploads(max_age_seconds: int = 3600):
    """Periodically purges temporary uploaded images older than max_age_seconds."""
    now = time.time()
    try:
        for file_path in config.UPLOADS_DIR.iterdir():
            if file_path.is_file() and not file_path.name.startswith("."):
                if now - file_path.stat().st_mtime > max_age_seconds:
                    file_path.unlink(missing_ok=True)
    except Exception as e:
        app.logger.warning(f"Error cleaning temporary uploads: {e}")


@app.context_processor
def inject_global_data():
    """Injects common metadata available across all Jinja2 templates."""
    model_ready = (
        predictor.is_model_loaded
        if predictor is not None
        else bool(INFERENCE_API_URL)
    )
    return {
        "app_title": "FloraGuard AI - Plant Disease Detection",
        "current_year": 2026,
        "is_model_ready": model_ready,
        "total_classes": (
            len(predictor.class_mapping)
            if predictor is not None and predictor.is_model_loaded
            else 38
        ),
    }


@app.route("/")
def index():
    """Landing Page: Project objectives, how it works, supported crops."""
    cleanup_old_uploads()
    return render_template("index.html")


@app.route("/detect", methods=["GET", "POST"])
def detect():
    """
    Disease Detection Page:
    - GET: Renders upload interface with sample leaf selectors.
    - POST: Receives uploaded image, performs validation, runs inference, and displays results.
    """
    if request.method == "POST":
        cleanup_old_uploads()

        # Check if file was provided
        if "file" not in request.files:
            flash("No file part in the submission. Please choose an image.", "warning")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No image file was selected. Please choose a leaf photograph.", "warning")
            return redirect(request.url)

        if not config.allowed_file(file.filename):
            flash(
                "Invalid file format! Supported formats are JPG, JPEG, PNG, and WEBP.",
                "danger",
            )
            return redirect(request.url)

        # Secure filename and prevent collisions with UUID prefix
        original_name = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex[:8]}_{int(time.time())}_{original_name}"
        saved_path = config.UPLOADS_DIR / unique_filename

        try:
            file.save(saved_path)

            # Perform prediction
            prediction_result = predict_image(saved_path)

            if VERCEL_DEPLOYMENT:
                mime_type = mimetypes.guess_type(original_name)[0] or "image/jpeg"
                encoded_image = base64.b64encode(saved_path.read_bytes()).decode("ascii")
                image_url = f"data:{mime_type};base64,{encoded_image}"
            else:
                image_url = url_for("uploaded_file", filename=unique_filename)
            result_page = render_template(
                "result.html",
                result=prediction_result,
                image_url=image_url,
                filename=original_name,
            )
            if VERCEL_DEPLOYMENT:
                saved_path.unlink(missing_ok=True)
            return result_page
        except Exception as e:
            if VERCEL_DEPLOYMENT:
                saved_path.unlink(missing_ok=True)
            app.logger.error(f"Prediction failed for {unique_filename}: {e}")
            flash(
                f"An error occurred while analyzing the image: {str(e)}. "
                "Please ensure the image is a valid, uncorrupted leaf photograph.",
                "danger",
            )
            return redirect(request.url)

    # GET request
    # Gather sample images if present in static/images/samples/
    sample_images = []
    if config.SAMPLES_DIR.exists():
        sample_images = [
            f.name
            for f in config.SAMPLES_DIR.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
        ]
    return render_template("detect.html", sample_images=sample_images)


@app.route("/detect/sample/<filename>")
def detect_sample(filename: str):
    """Quick diagnostic test using one of the bundled sample leaf images."""
    safe_name = secure_filename(filename)
    sample_path = config.SAMPLES_DIR / safe_name
    if not sample_path.exists():
        flash("Sample leaf image not found.", "warning")
        return redirect(url_for("detect"))

    try:
        prediction_result = predict_image(sample_path)
        image_url = url_for("static", filename=f"images/samples/{safe_name}")
        return render_template(
            "result.html",
            result=prediction_result,
            image_url=image_url,
            filename=safe_name,
        )
    except Exception as e:
        flash(f"Error analyzing sample image: {e}", "danger")
        return redirect(url_for("detect"))


@app.route("/about")
def about():
    """About Page: Deep learning architecture, methodology, dataset, and limitations."""
    return render_template("about.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for programmatic leaf disease prediction."""
    expected_token = os.environ.get("PREDICTION_API_TOKEN", "")
    authorization = request.headers.get("Authorization", "")
    if expected_token and not hmac.compare_digest(
        authorization, f"Bearer {expected_token}"
    ):
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file field provided in request"}), 400

    file = request.files["file"]
    if file.filename == "" or not config.allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid or missing file format"}), 400

    unique_filename = f"api_{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
    saved_path = config.UPLOADS_DIR / unique_filename

    try:
        file.save(saved_path)
        result = predict_image(saved_path)
        # Delete file after API processing
        saved_path.unlink(missing_ok=True)
        return jsonify(result), 200
    except Exception as e:
        saved_path.unlink(missing_ok=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/uploads/<filename>")
def uploaded_file(filename: str):
    """Safely serves uploaded leaf photos."""
    safe_name = secure_filename(filename)
    return send_from_directory(app.config["UPLOAD_FOLDER"], safe_name)


@app.errorhandler(413)
def request_entity_too_large(error):
    max_size_mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    flash(f"Uploaded file is too large! Maximum allowed image size is {max_size_mb} MB.", "danger")
    return redirect(url_for("detect")), 413


@app.errorhandler(404)
def page_not_found(error):
    return render_template("base.html", error_title="404 - Page Not Found", error_msg="The page you requested could not be located."), 404


@app.errorhandler(500)
def internal_server_error(error):
    return render_template("base.html", error_title="500 - Server Error", error_msg="An internal processing error occurred."), 500


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("[FLASK] PLANT DISEASE DETECTION WEB APPLICATION")
    print("Serving on: http://127.0.0.1:5000")
    model_ready = predictor is not None and predictor.is_model_loaded
    print(f"Model Status: {'LOADED & READY' if model_ready else 'TRAINING REQUIRED (Fallback Mode Active)'}")
    print("=" * 70 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
