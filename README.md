# Plant Disease Detection Using Deep Learning and CNN

> **FloraGuard AI**: A Production-Grade Foliar Pathology Classification System Powered by MobileNetV2 Transfer Learning, Computer Vision Preprocessing, and a Full-Stack Flask Web Application.

---

## Abstract

Agriculture is the foundation of global food security and economic stability. Foliar infections and insect infestations cause substantial annual reductions in crop yields, often between 20% and 40%. Rapid and accurate pathogen detection is critical to arrest disease progression before irreversible damage occurs. 

This project implements an automated, computer vision-based plant disease detection system titled **"Plant Disease Detection Using Deep Learning and CNN"**. Utilizing the **PlantVillage** dataset, we implement a transfer learning architecture based on **MobileNetV2** pretrained on ImageNet. The model classifies healthy and diseased leaf specimens across 38 distinct plant-pathogen categories with high accuracy and low latency. The deep learning inference engine is integrated into a modern, responsive **Flask** web application that provides:
1. Drag-and-drop foliar image upload with real-time browser preview.
2. Prediction confidence scores and top-3 candidate distribution analysis.
3. Structured, actionable agricultural disease management guidance (Cultural, Organic, Chemical, and Preventive).
4. Full model evaluation suite including loss/accuracy curves, Scikit-learn classification reports, and Seaborn confusion matrices.

---

## Problem Statement

Traditional agricultural disease surveillance relies on manual visual examination by experienced agronomists or laboratory microbiological diagnostics (such as PCR or fungal culture plating). These methods suffer from:
- **Severe Scarcity of Experts:** Smallholder farming communities lack immediate access to certified plant pathologists.
- **Diagnostic Delays:** Sending foliar tissue samples to distant diagnostic laboratories takes days or weeks, allowing disease spread.
- **Misuse of Agrochemicals:** Inaccurate symptom recognition leads farmers to apply inappropriate broad-spectrum pesticides, inducing chemical resistance, inflating production costs, and causing environmental toxicity.

An automated, non-invasive, accessible, and mathematically validated deep learning diagnostic tool is required to bridge the diagnostic gap directly in the field.

---

## Objectives

- **Automated Leaf Classification:** Identify and classify 38 plant disease categories across crops including Tomato, Potato, Corn, Apple, Grape, Pepper, and others.
- **Lightweight Deep Transfer Learning:** Implement MobileNetV2 with depthwise separable convolutions to balance classification accuracy, inference speed, and compact model size suitable for future edge deployment.
- **Data Augmentation & Regularization:** Mitigate overfitting using geometric transformations (rotation, translation, zoom, flips) and dropout regularization.
- **Comprehensive Evaluation:** Compute overall test accuracy, Top-3 accuracy, class-level precision, recall, F1-scores, and generate normalized confusion matrix heatmaps.
- **Actionable Agronomic Guidance:** Supply structured, realistic disease management practices alongside class predictions.
- **Full-Stack Web Interface:** Deploy an agricultural dashboard utilizing Flask, Bootstrap 5, and Vanilla CSS with robust validation and security.

---

## Existing System vs. Proposed System

| Feature / Dimension | Conventional / Existing Systems | Proposed FloraGuard AI System |
| :--- | :--- | :--- |
| **Diagnostic Method** | Manual visual inspection or lab assays | Deep Convolutional Neural Network (MobileNetV2) |
| **Response Time** | Days to weeks | Real-time (< 100 milliseconds inference) |
| **Human Bias** | Subjective, prone to human fatigue | Objective, reproducible probabilistic outputs |
| **Crop Coverage** | Typically limited to 1 or 2 specific crops | 38 plant-pathogen categories spanning 14 species |
| **Output Information** | Binary or qualitative notes | Top prediction, Top-3 distribution, confidence meter, symptoms, and 4-tier management advice |
| **Hardware Overhead** | Heavy laboratory microscopes / PCR gear | Runs on standard CPU/GPU hardware; edge-ready |
| **User Interface** | Paper reports or complex lab software | Modern, mobile-responsive Flask web application |

---

## System Architecture

```mermaid
graph TD
    A([Farmer / Student / User]) -->|Uploads Leaf Image| B[Web Browser / UI]
    B -->|HTTP POST Request| C[Flask Server app.py]
    
    subgraph Computer Vision & Preprocessing Pipeline
        C -->|Format & Size Validation| D[Image Validator]
        D -->|Safe Open & RGB Conversion| E[Pillow / OpenCV]
        E -->|Resize to 224x224| F[Tensor Normalizer]
        F -->|MobileNetV2 Preprocessing [-1, 1]| G[Batch Tensor 1, 224, 224, 3]
    end

    subgraph Deep Learning Inference Engine
        G --> H[Trained MobileNetV2 Model .keras]
        H -->|Softmax Probability Vector| I[Prediction Post-Processor]
        I -->|Top-1 Class & Confidence Score| J[Primary Diagnosis]
        I -->|Top-3 Probability Ranks| K[Top-3 Breakdown]
    end

    subgraph Agricultural Knowledge Base
        J --> L[(disease_details.json)]
        L --> M[Symptoms & Biological Details]
        L --> N[Cultural & Organic Guidance]
        L --> O[Chemical & Preventive Tips]
    end

    M & N & O & J & K --> P[Result Renderer]
    P -->|Renders result.html| B
```

---

## Dataset Description

The project is built on the benchmark **PlantVillage** dataset, widely acknowledged as the standard benchmark in plant pathology computer vision research.

- **Total Images:** ~54,305 labeled leaf photographs.
- **Total Categories:** 38 distinct classes (comprising diseased conditions and healthy controls).
- **Supported Crops:**
  - **Tomato:** Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy.
  - **Potato:** Early Blight, Late Blight, Healthy.
  - **Corn (Maize):** Cercospora Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy.
  - **Apple:** Apple Scab, Black Rot, Cedar Apple Rust, Healthy.
  - **Bell Pepper:** Bacterial Spot, Healthy.
  - **Grape:** Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis), Healthy.
  - **Others:** Peach (Bacterial Spot, Healthy), Cherry (Powdery Mildew, Healthy), Strawberry (Leaf Scorch, Healthy), Squash (Powdery Mildew), Orange (Citrus Greening), Blueberry, Raspberry, Soybean.
- **Partitioning Strategy:**
  - **Training Set (70%):** Model parameter optimization with data augmentation.
  - **Validation Set (15%):** Hyperparameter tuning, early stopping, learning rate adaptation.
  - **Held-Out Test Set (15%):** Unbiased final generalization evaluation.

### Dataset Download Instructions (Kaggle)

1. **Via Kaggle CLI:**
   ```bash
   kaggle datasets download -d emmarex/plantdisease
   unzip plantdisease.zip -d dataset/
   ```
2. **Via Kaggle Web Portal:**
   - Visit: [Kaggle PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) or [PlantVillage New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
   - Click **Download (1.8 GB)**.
   - Extract the folder contents into the `plant-disease-detection/dataset/` directory.

---

## Deep Learning Model Architecture

The core classifier utilizes **MobileNetV2** with transfer learning:

```
Input Image (224 x 224 x 3)
      │
[MobileNetV2 Preprocessing: Scales pixels to range [-1, 1]]
      │
[Pretrained MobileNetV2 Base (ImageNet Weights, Freeze Base)]
  ├── Inverted Residual Blocks (x17)
  ├── Depthwise Separable Convolutions (3x3 depthwise + 1x1 pointwise)
  └── Linear Bottleneck Layers
      │
Global Average Pooling 2D (1280 feature vectors)
      │
Batch Normalization Layer
      │
Dropout (Rate = 0.3)
      │
Dense Layer (256 Neurons, ReLU Activation)
      │
Batch Normalization Layer
      │
Dropout (Rate = 0.2)
      │
Dense Output Layer (38 Neurons, Softmax Activation)
```

### Why MobileNetV2?
1. **Depthwise Separable Convolutions:** Standard convolutions perform channel and spatial filtering simultaneously. MobileNetV2 decouples this into a depthwise convolution (spatial filtering per channel) and a $1 \times 1$ pointwise convolution (linear combination across channels), reducing compute by approximately:
   $$\frac{\text{Cost}_{\text{Depthwise Separable}}}{\text{Cost}_{\text{Standard}}} = \frac{1}{N} + \frac{1}{D_k^2} \approx \frac{1}{9}$$
2. **Inverted Residuals & Linear Bottlenecks:** Information flows through narrow bottlenecks with skip connections between low-dimensional manifolds, avoiding non-linear information loss (manifold of interest principle).
3. **Low Parameter Footprint:** 3.5 million parameters compared to 138 million in VGG16, ensuring rapid inference and minimal RAM consumption.

---

## Training Methodology

- **Loss Function:** Categorical Cross-Entropy:
  $$\mathcal{L} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$
- **Optimizer:** Adam ($\beta_1 = 0.9, \beta_2 = 0.999$, Initial Learning Rate = $1 \times 10^{-3}$).
- **Data Augmentation:** Real-time on-the-fly random rotations ($\pm 15^\circ$), horizontal and vertical flips, random zoom ($\pm 15\%$), and spatial translations ($\pm 10\%$).
- **Regularization:**
  - Dropout ($p = 0.3$ and $p = 0.2$)
  - Batch Normalization
  - Early Stopping with patience = 5 monitoring `val_loss` (restores best weights)
  - Learning Rate Plateau Reduction: Drops LR by factor of 0.2 when `val_loss` plateaus for 3 epochs.
- **Fine-Tuning:** Optional Phase 2 unfreezes upper MobileNetV2 layers (from layer 100 onwards) with a low learning rate ($1 \times 10^{-5}$) to specialize top convolutional filters.

---

## Evaluation Metrics

The model is evaluated on the held-out test split using standard classification metrics:

1. **Accuracy:**
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
2. **Precision:**
   $$\text{Precision} = \frac{TP}{TP + FP}$$
3. **Recall (Sensitivity):**
   $$\text{Recall} = \frac{TP}{TP + FN}$$
4. **F1-Score (Harmonic Mean):**
   $$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
5. **Confusion Matrix:** Generates an $N \times N$ matrix displaying true positive hits alongside off-diagonal classification confusion.

*All measured metrics are saved to `results/evaluation_metrics.json` and plotted in `results/confusion_matrix.png` and `results/training_history.png`.*

---

## Project Folder Structure

```
plant-disease-detection/
│
├── app.py                     # Flask web server, routes, upload validation, error handlers
├── train.py                   # MobileNetV2 transfer learning training pipeline
├── predict.py                 # Core inference engine with image preprocessing & caching
├── evaluate.py                # Model evaluation, confusion matrix & classification report
├── config.py                  # Central configuration (paths, hyperparameters, constants)
├── requirements.txt           # Lightweight Vercel web dependencies
├── requirements-inference.txt # Render TensorFlow prediction dependencies
├── requirements-training.txt  # Full local development and training dependencies
├── vercel.json                # Vercel serverless routing configuration
├── render.yaml                # Render inference service blueprint
├── api/index.py               # Vercel Flask function entry point
├── README.md                  # Comprehensive documentation, setup guide & viva Q&A
├── .gitignore                 # Excludes models, uploads, dataset, pycache
│
├── models/                    # Saved weights & class mappings
│   ├── .gitkeep
│   ├── class_names.json       # Exact class index mapping
│   └── plant_disease_model.keras # Trained MobileNetV2 model
│
├── dataset/                   # Dataset directory (PlantVillage structure)
│   ├── .gitkeep
│   ├── train/                 # (Generated / populated by splits or user)
│   ├── val/
│   └── test/
│
├── results/                   # Evaluation artifacts & training plots
│   ├── .gitkeep
│   ├── training_history.png   # Accuracy & Loss curves
│   ├── confusion_matrix.png   # Normalized confusion matrix heatmap
│   └── evaluation_metrics.json# Precision, recall, f1-score & accuracy metrics
│
├── disease_info/
│   └── disease_details.json   # 38 plant classes: crop, disease, symptoms, management
│
├── scripts/
│   └── prepare_samples.py     # Generates sample leaf images and mini dev dataset
│
├── tests/
│   └── test_pipeline.py       # Integration unit test suite for routes and pipeline
│
├── templates/                 # Jinja2 HTML templates
│   ├── base.html              # Base layout with agricultural navbar & footer
│   ├── index.html             # Landing page with workflow & supported crops
│   ├── detect.html            # Upload leaf page with drag-and-drop & live preview
│   ├── result.html            # Detailed prediction results, confidence & guidance
│   └── about.html             # Project architecture, methodology, dataset info
│
├── static/
│   ├── css/
│   │   └── style.css          # Custom agricultural green-and-white theme
│   ├── js/
│   │   └── script.js          # Live image preview, loading spinner, drag-and-drop
│   └── images/
│       └── samples/           # Sample test images for immediate demonstration
│
└── uploads/                   # Temporary folder for user-uploaded leaf images
    └── .gitkeep
```

---

## Installation & Setup Instructions (Windows / VS Code)

### Prerequisites
- Python 3.10 - 3.13 installed on Windows.
- Git (optional).
- Visual Studio Code or PyCharm.

### Step 1: Open Terminal in VS Code
Open VS Code, press ``Ctrl + ` `` to open the integrated PowerShell terminal, and navigate to the project directory:
```powershell
cd plant-disease-detection
```

### Step 2: Create and Activate Virtual Environment
```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# (If PowerShell displays execution policy error, run once:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser)
```

### Step 3: Install Required Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt
```

### Step 4: Generate Sample Images & Quick Dev Setup
Run the bundled asset generator to populate sample leaf images and initialize the development data structure:
```powershell
python scripts\prepare_samples.py
```

---

## Model Training & Execution Instructions

### Option A: Quick Development / Demo Training (1-2 minutes)
To test and verify the entire training pipeline end-to-end without waiting for the 2GB dataset:
```powershell
python train.py --quick-dev
```
This trains MobileNetV2 on the mini dataset, saves `models/plant_disease_model.keras`, records `models/class_names.json`, and outputs `results/training_history.png`.

### Option B: Full Training on PlantVillage Dataset
Once you have placed the full PlantVillage dataset inside the `dataset/` directory:
```powershell
# Standard Transfer Learning (15 Epochs)
python train.py --dataset dataset --epochs 15 --batch-size 32

# With Phase 2 Fine-Tuning enabled
python train.py --dataset dataset --epochs 15 --fine-tune
```

---

## Model Evaluation Instructions

Run the evaluation module on the test dataset:
```powershell
python evaluate.py
```
This will:
1. Load `models/plant_disease_model.keras` and `models/class_names.json`.
2. Evaluate test images.
3. Print the complete Scikit-learn Classification Report (Precision, Recall, F1).
4. Save the normalized Confusion Matrix heatmap to `results/confusion_matrix.png`.
5. Write all numerical evaluation metrics to `results/evaluation_metrics.json`.

---

## Running the Web Application

Launch the Flask development server:
```powershell
python app.py
```
You will see:
```
======================================================================
🌿 FLASK PLANT DISEASE DETECTION WEB APPLICATION
🚀 Serving on: http://127.0.0.1:5000
🔍 Model Status: LOADED & READY
======================================================================
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## Deploying to Vercel

The TensorFlow runtime is too large for a Vercel serverless function. This project
deploys the lightweight Flask website to Vercel and forwards image predictions
to a separate Render service running the model.

1. Push the project folder to a Git repository. Ensure
   `models/plant_disease_model.keras` and `models/class_names.json` are included;
   the model weights are allowed by `.gitignore`.
2. In Render, create a new **Blueprint** from the repository. Render reads
   `render.yaml`, installs `requirements-inference.txt`, and starts the inference
   service. The service needs the trained model file in the repository.
3. After the Render service is live, copy its public URL and the generated
   `PREDICTION_API_TOKEN` from its environment settings.
4. Import the same repository into Vercel, using the project folder as the root
   directory. Vercel uses `vercel.json` and the lightweight `requirements.txt`.
5. Add these Vercel environment variables:
   - `INFERENCE_API_URL`: the Render service base URL, such as
     `https://floraguard-inference.onrender.com` (no trailing slash).
   - `INFERENCE_API_TOKEN`: the same value as Render's `PREDICTION_API_TOKEN`.
   - `FLASK_SECRET_KEY`: a long, random secret value.
6. Redeploy the Vercel project after setting the environment variables.

Vercel stores uploads in temporary `/tmp` storage and limits image uploads to
3 MB so the result page can display the uploaded image within serverless request
and response limits. The Render service uses a paid standard instance in the
provided blueprint to give TensorFlow enough memory; choose an appropriate
instance size based on your provider's current limits and pricing.

For local development, continue to install `requirements-training.txt`; it
contains the complete TensorFlow, image-processing, evaluation, and training
dependencies. The Render service only installs the smaller
`requirements-inference.txt` runtime set.

### Testing the Web Interface
1. Click **Detect Disease** in the top navigation.
2. Either drag and drop a leaf image into the upload zone or click on one of the **Quick Test Sample Leaf Images** at the bottom (e.g. `Tomato_Early_Blight.jpg` or `Tomato_Healthy.jpg`).
3. Click **Predict Disease**.
4. Observe:
   - Identified Crop Name and Disease Name.
   - Model Confidence percentage and confidence rating badge.
   - Top-3 candidate predictions with probability progress bars.
   - Biological pathological description.
   - Foliar symptoms checklist.
   - Four-tier agricultural disease management recommendations.

---

## Running the Automated Test Suite

To run integration tests verifying routes, image preprocessing, and configuration:
```powershell
python -m unittest discover -s tests -p test_pipeline.py
```

---

## Limitations & Future Scope

### Limitations
- **Background Bias:** PlantVillage images are captured in lab settings with neutral backgrounds. Heavy field clutter (soil, weeds, shadows) may slightly lower confidence.
- **Multiple Concurrent Pathologies:** The model classifies the primary dominant symptom per leaf and does not model secondary co-infections.
- **Out-of-Distribution Inputs:** Submitting non-leaf photos will map to the closest mathematical embedding unless flagged by low confidence.

### Future Scope
- **Edge Deployment (TFLite & Flutter):** Quantize the model (INT8/FP16) and deploy to native Android/iOS mobile apps for offline field use.
- **Grad-CAM Explainability:** Render Gradient-weighted Class Activation Mapping overlays highlighting the exact lesion areas that triggered the classification.
- **Drone Aerial Diagnostics:** Integrate aerial RGB/Multispectral drone imagery for farm-wide canopy health monitoring.

---

## References

1. **Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C.** (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
2. **Hughes, D., & Salathé, M.** (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv preprint arXiv:1511.08060 (PlantVillage).
3. **Mohanty, S. P., Hughes, D. P., & Salathé, M.** (2016). *Using deep learning for image-based plant disease detection.* Frontiers in Plant Science, 7, 1419.
4. **Chollet, F.** (2017). *Xception: Deep Learning with Depthwise Separable Convolutions.* CVPR 2017.

---

## Viva Voce Questions & Answers (Comprehensive Exam Preparation)

### 1. What is the core objective of this project?
> **Answer:** To develop an automated, end-to-end deep learning system that classifies foliar plant diseases across 38 categories from leaf photographs using a Convolutional Neural Network (MobileNetV2) and provides confidence scores and agricultural management recommendations via a Flask web application.

### 2. Why is MobileNetV2 preferred over heavier architectures like VGG16 or ResNet50?
> **Answer:** MobileNetV2 uses **depthwise separable convolutions** and **inverted residuals**, having only ~3.5 million parameters compared to VGG16's ~138 million. It provides comparable accuracy on fine-grained image tasks while consuming drastically lower memory and computational power (FLOPs), enabling sub-100ms inference on standard CPUs and making it suitable for edge mobile deployment.

### 3. Explain how Depthwise Separable Convolution works.
> **Answer:** It splits standard convolution into two separate steps:
> 1. **Depthwise Convolution:** Applies a single spatial convolutional filter ($3 \times 3$) per input channel independently.
> 2. **Pointwise Convolution:** Applies a $1 \times 1$ convolution across all channels to linearly combine the features.
> This reduces computation by a factor of $\frac{1}{N} + \frac{1}{D_k^2} \approx \frac{1}{9}$ for a $3 \times 3$ kernel, drastically accelerating execution.

### 4. What are Inverted Residuals and Linear Bottlenecks?
> **Answer:** Standard residual networks (ResNet) connect high-dimensional feature blocks with bottleneck layers in between. MobileNetV2 inverts this: the skip connection connects low-dimensional bottlenecks directly, while expanding the intermediate layer to high dimensions. The bottleneck layers use linear activations instead of ReLU to avoid discarding manifold information (the "manifold of interest" principle).

### 5. What is Transfer Learning and why did you use it?
> **Answer:** Transfer learning leverages feature representations (edges, textures, shapes) learned by a model trained on a large benchmark dataset (ImageNet-1K) and transfers them to a target domain (PlantVillage). This dramatically reduces training time, requires fewer training epochs, and prevents overfitting on limited domain data.

### 6. What preprocessing was applied to images before feeding into MobileNetV2?
> **Answer:** The image is converted to RGB, resized to $224 \times 224$ pixels, converted to a float32 tensor, and normalized using `mobilenet_v2.preprocess_input`, which maps raw $[0, 255]$ pixel values to the interval $[-1, 1]$:
> $$\text{Pixel}_{\text{norm}} = \frac{\text{Pixel}}{127.5} - 1.0$$

### 7. How does Data Augmentation prevent overfitting?
> **Answer:** It synthetically increases dataset diversity by applying random rotations, horizontal/vertical flips, zoom, and spatial translations. The network sees slightly varied geometric variations in each epoch, forcing it to learn rotation-invariant and position-invariant pathological features rather than memorizing individual image pixels.

### 8. What is the role of Global Average Pooling (GAP) versus Flattening?
> **Answer:** Flattening preserves all spatial coordinates, generating huge parameter matrices that lead to severe overfitting. Global Average Pooling computes the average value of each feature map across the spatial dimensions ($H \times W$), reducing each feature map to a single scalar. This reduces parameter count drastically and acts as structural regularizer.

### 9. Why did you use Categorical Cross-Entropy loss?
> **Answer:** The task is multi-class classification where each leaf belongs to one of $C$ mutually exclusive classes. Categorical Cross-Entropy computes the negative log-likelihood of the true class probability:
> $$\mathcal{L} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$
> It heavily penalizes confident incorrect predictions and drives the network toward zero error.

### 10. Explain Early Stopping and Learning Rate Reduction on Plateau.
> **Answer:** 
> - **Early Stopping:** Monitors validation loss; if it does not improve for 5 consecutive epochs, training terminates and the best model weights are restored.
> - **ReduceLROnPlateau:** If validation loss stagnates for 3 epochs, the learning rate is scaled down by a factor of 0.2 (divided by 5), allowing the optimizer to take finer gradient descent steps into narrow minima.

### 11. What is the difference between Precision and Recall in this project?
> **Answer:**
> - **Precision:** Out of all leaves predicted as disease $X$, how many actually had disease $X$? (Minimizes false alarms).
> - **Recall (Sensitivity):** Out of all leaves that truly had disease $X$, how many did the model correctly identify? (Critical in agriculture: missing a disease allows outbreaks to destroy entire fields).

### 12. What does the Confusion Matrix show?
> **Answer:** An $N \times N$ matrix where row $i$ represents true class instances and column $j$ represents predicted class instances. Diagonal cells represent correct classifications ($TP$), while off-diagonal elements expose specific confusion (e.g., misclassifying Early Blight as Late Blight).

### 13. How does the Flask web app prevent model reload overhead?
> **Answer:** The `PlantDiseasePredictor` is designed as a **singleton**. The trained Keras model and JSON class mappings are loaded into memory once during application startup. Subsequent HTTP requests reuse the existing model instance in memory, achieving instant inference.

### 14. How does the system handle security on user-uploaded files?
> **Answer:** 
> 1. Restricts file extensions strictly to allowed image types (`.jpg`, `.jpeg`, `.png`, `.webp`).
> 2. Uses `werkzeug.utils.secure_filename` to prevent path traversal attacks (`../`).
> 3. Enforces a maximum payload size of 16 MB (`MAX_CONTENT_LENGTH`).
> 4. Periodically cleans up temporary files in `uploads/`.

### 15. What happens if a user uploads a low-confidence or non-leaf image?
> **Answer:** The system checks the highest prediction confidence score against a threshold (60%). If the confidence is below 60%, the UI displays a prominent diagnostic caution warning advising the user that the prediction may be unreliable, and recommending that they consult an agricultural extension specialist or provide a clearer photo.
