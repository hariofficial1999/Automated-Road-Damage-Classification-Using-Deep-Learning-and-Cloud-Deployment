# Automated Road Damage Classification Using Deep Learning and Cloud Deployment

This project classifies road damage into 3 categories — **Pothole**, **Crack**, and **Manhole** — using deep learning models. It also uses **Grad-CAM** to show which part of the image the model looked at, and gives repair recommendations.

---

## Overview

Checking roads manually for damage takes a lot of time and effort. This project uses **CNN** and **Transfer Learning** to classify road images taken from phones or vehicle cameras. A **Streamlit web app** is included for easy use.

---

## Features

| Feature | Description |
|---|---|
| **Classification** | Classifies into 3 types: Pothole, Crack, Manhole |
| **Grad-CAM** | Shows heatmap of which area the model focused on |
| **Recommendations** | Gives repair suggestions based on damage type |
| **Transfer Learning** | Fine-tuned **ResNet50** for better accuracy |
| **Web App** | Built with **Streamlit** |
| **Model Comparison** | Compared Baseline CNN, MobileNetV2, and ResNet50 |

---

## Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python |
| **Deep Learning** | TensorFlow, Keras |
| **Image Processing** | OpenCV, PIL, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Explainability** | Grad-CAM |
| **Data Handling** | Pandas, Scikit-learn |
| **Models** | Baseline CNN, MobileNetV2, ResNet50 |
| **Deployment** | Streamlit |

---

## Approach

### 1. Data Loading
- **Train Directory**: `data/train/` — with class folders (`0_Pothole`, `1_Crack`, `2_Manhole`)
- **Test Directory**: `data/test/` — same structure
- Loaded using `tf.keras.utils.image_dataset_from_directory`
- **Validation Split**: 20% from training data
- **Seed**: 123 for reproducibility

### 2. Data Preprocessing
- **Image Size**: Resized to `224×224`
- **Batch Size**: 32
- **Normalization**: Pixel values scaled to `[0, 1]` using `Rescaling(1./255)`
- **Pipeline Optimization**: Used `cache()`, `shuffle(1000)`, and `prefetch(AUTOTUNE)` for faster loading

### 3. Data Augmentation
Applied using `tf.keras.Sequential`:
```python
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])
```

### 4. Class Imbalance Handling
- Checked class distribution using sample counts
- Calculated `class_weight` using formula: `total / (NUM_CLASSES * count_per_class)`
- If `max_count / min_count > 2` → imbalance detected
- Applied `class_weight` during training to handle imbalance

---

## Models

### Model 1: Baseline CNN

Built from scratch using Sequential API:

| Layer | Details |
|---|---|
| Input | `224×224×3` |
| Rescaling | Normalize to `[0, 1]` |
| Data Augmentation | Flip, Rotation, Zoom, Contrast |
| Conv2D + MaxPool | 32 filters, kernel size 3 |
| Conv2D + MaxPool | 64 filters, kernel size 3 |
| Dense | Fully connected layers |
| Output | 3 classes (softmax) |

- **Optimizer**: Adam (`lr=1e-3`)
- **Loss**: `sparse_categorical_crossentropy`
- **Epochs**: 10
- **Result**: ~48% accuracy
- **Saved as**: `baseline_cnn.h5`

---

### Model 2: MobileNetV2 (Transfer Learning)

**Step 1 — Load Pretrained Model**:
- `MobileNetV2(include_top=False, weights="imagenet")`
- Initially all layers frozen (`trainable=False`)

**Step 2 — Add Custom Head**:
```python
mobilenet_model = models.Sequential([
    Input(224, 224, 3),
    data_augmentation,
    Lambda(mobilenet_v2.preprocess_input),
    base_mobilenet,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation="softmax")
])
```

**Step 3 — Initial Training**:
- Optimizer: Adam (`lr=1e-4`), Epochs: 8

**Step 4 — Fine-Tuning**:
- Unfroze last 50 layers
- Kept BatchNormalization layers trainable
- Re-compiled with lower learning rate: Adam (`lr=1e-5`)
- Fine-tuned for 10 more epochs

| Fine-Tune Detail | Value |
|---|---|
| Total Layers | 154 |
| Trainable Layers | 85 |
| Non-trainable Layers | 69 |

- **Result**: ~56% accuracy
- **Saved as**: `mobilenetv2_road_damage.h5`

---

### Model 3: ResNet50 (Transfer Learning) — Best Model

**Step 1 — Load Pretrained Model**:
- `ResNet50(include_top=False, weights="imagenet")`
- Initially all layers frozen

**Step 2 — Add Custom Head**:
```python
resnet_model = models.Sequential([
    Input(224, 224, 3),
    data_augmentation,
    Lambda(resnet50.preprocess_input),
    base_resnet,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation="softmax")
])
```

**Step 3 — Initial Training**:
- Optimizer: Adam (`lr=1e-4`), Epochs: 8

**Step 4 — Fine-Tuning**:
- Unfroze last 30 layers
- Kept BatchNormalization layers trainable
- Re-compiled with: Adam (`lr=1e-5`)
- Fine-tuned for 10 more epochs

- **Result**: ~64% accuracy (Best)
- **Saved as**: `resnet50_fine_tuned.h5`

---

## Evaluation

### Accuracy Comparison

| Model | Test Accuracy |
|---|---|
| Baseline CNN | ~48% |
| MobileNetV2 (Fine-Tuned) | ~56% |
| **ResNet50 (Fine-Tuned)** | **~64%** |

### Evaluation Metrics Used
- **Accuracy** — overall correct predictions
- **Precision** — how many predicted positives are actually correct
- **Recall** — how many actual positives are correctly predicted
- **F1-Score** — balance between precision and recall
- **Classification Report** — per-class breakdown with styled table (Blues gradient)
- **Confusion Matrix** — heatmap showing correct vs misclassified predictions

### Grad-CAM (Explainability)
- Applied on the best model (ResNet50)
- Generates heatmap overlay on the input image
- Shows which region of the road the model focused on for its prediction

---

## Project Structure

```text
Final Project/
├── data/
│   ├── images/                          # Raw dataset images
│   ├── labels/                          # Annotations
│   ├── train/                           # Training data
│   │   ├── 0_Pothole/
│   │   ├── 1_Crack/
│   │   └── 2_Manhole/
│   └── test/                            # Test data
│       ├── 0_Pothole/
│       ├── 1_Crack/
│       └── 2_Manhole/
├── TEST.ipynb                           # Full Training & Evaluation Notebook
├── app.py                               # Streamlit Web App
├── baseline_cnn.h5                      # Baseline CNN Model (~48%)
├── mobilenetv2_road_damage.h5           # MobileNetV2 Fine-Tuned Model (~56%)
├── resnet50_fine_tuned.h5               # ResNet50 Fine-Tuned Model (~64%) — Best
├── Final Project_B94.pdf                # Project Report
└── README.md                            # This file
```

---

## How to Run

### Install Dependencies
```bash
pip install tensorflow streamlit numpy pillow opencv-python matplotlib seaborn scikit-learn pandas
```

### 1. Train the Models
```bash
jupyter notebook TEST.ipynb
```
Run all cells. This will:
- Train Baseline CNN (10 epochs)
- Train MobileNetV2 (8 epochs + 10 fine-tune epochs)
- Train ResNet50 (8 epochs + 10 fine-tune epochs)
- Save all 3 `.h5` model files
- Generate classification reports and confusion matrices

### 2. Run the Web App
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`. Upload a road image to get the prediction.

---

## Web App

1. **Upload** a road image (JPG, JPEG, PNG)
2. **Prediction** — Shows damage type (Pothole / Crack / Manhole)
3. **Confidence Score** — Shows how confident the model is
4. **Class Probabilities** — Shows probability for all 3 classes
5. **Recommendation** — Gives repair suggestion

| Damage Type | Recommendation |
|---|---|
| Pothole | Schedule immediate repair |
| Crack | Monitor and seal the crack soon |
| Manhole | Inspect cover condition and alignment |

---

## Notebook Workflow (TEST.ipynb)

The notebook has 107 cells and follows this flow:

| Step | Section | Cells |
|---|---|---|
| 1 | Imports & Setup | Cell 0–2 |
| 2 | Data Loading (Train/Val/Test) | Cell 3–20 |
| 3 | Data Pipeline Optimization | Cell 21–23 |
| 4 | Data Augmentation & Normalization | Cell 24–31 |
| 5 | Class Imbalance Check | Cell 32–36 |
| 6 | Baseline CNN (Build, Train, Evaluate, Save) | Cell 37–52 |
| 7 | MobileNetV2 (Load, Train, Fine-Tune, Evaluate, Save) | Cell 53–78 |
| 8 | ResNet50 (Load, Train, Fine-Tune, Evaluate, Save) | Cell 79–103 |
| 9 | Model Comparison & Final Chart | Cell 104–106 |

---

## Future Work

- Connect with live camera for real-time detection
- Add GPS to track damage locations on a map
- Build a mobile app for field inspections
- Add more damage classes (speed bumps, lane markings)
- Deploy on cloud (AWS / GCP / Azure)

---

## Author

**Hariharan** — Data Science Intern  
Project: Automated Road Damage Classification Using Deep Learning and Cloud Deployment

---

