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

### 1. Data Preparation
- **Dataset**: Road Damage Dataset (RDD) — 3 classes (Pothole, Crack, Manhole)
- **Preprocessing**: Resized to `224×224`, normalized (`/255.0`)
- **Augmentation**: Rotation, flips, brightness change, blur
- **Class Imbalance**: Handled using `class_weights`
- **Split**: Train / Test

### 2. Model Development

| Model | Type | Description |
|---|---|---|
| **Baseline CNN** | Custom CNN | Built from scratch using Conv2D, MaxPool, Dense |
| **MobileNetV2** | Transfer Learning | Fine-tuned last 50 layers + BatchNorm |
| **ResNet50** | Transfer Learning | Fine-tuned, best accuracy |

#### Fine-Tuning Code:
```python
# Step 1: Unfreeze base model
base_model.trainable = True

# Step 2: Freeze first layers
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Step 3: Keep BatchNormalization trainable
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True
```

### 3. Model Evaluation

| Model | Accuracy | Status |
|---|---|---|
| Baseline CNN | ~51% | Weak |
| MobileNetV2 (Fine-Tuned) | ~54% | Moderate |
| **ResNet50 (Fine-Tuned)** | **~61%** | **Best Model** |

- ResNet50 gave the best accuracy and F1-score
- Evaluated using Confusion Matrix, Classification Report, and Grad-CAM

### 4. Grad-CAM
- Generates heatmaps on top of the original image
- Shows which part of the image the model focused on
- Useful for understanding model predictions

---

## Project Structure

```text
Final Project/
├── data/
│   ├── images/                          # Dataset images
│   ├── labels/                          # Annotations
│   ├── train/                           # Training data
│   └── test/                            # Test data
├── TEST.ipynb                           # Training & Evaluation Notebook
├── app.py                               # Streamlit Web App
├── test_gradcam.py                      # Grad-CAM Testing Script
├── baseline_cnn.h5                      # Baseline CNN Model
├── mobilenetv2_road_damage.h5           # MobileNetV2 Model
├── resnet50_fine_tuned.h5               # ResNet50 Model (Best)
├── Final Project_B94.pdf                # Project Report
└── README.md                            # This file
```

---

## How to Run

### Install Dependencies
```bash
pip install tensorflow streamlit numpy pillow opencv-python matplotlib seaborn scikit-learn pandas
```

### 1. Train the Model
```bash
jupyter notebook TEST.ipynb
```
Run all cells. This will generate the `.h5` model files.

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

## Results

| Metric | Baseline CNN | MobileNetV2 FT | ResNet50 FT |
|---|---|---|---|
| **Accuracy** | ~51% | ~54% | **~61%** |
| **Best For** | Baseline comparison | Lightweight use | Best accuracy |
| **Layers Fine-Tuned** | All (custom) | Last 50 + BatchNorm | Last layers + BatchNorm |

ResNet50 gave the best results compared to other models.

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

## License

This project is for educational and research purposes.
