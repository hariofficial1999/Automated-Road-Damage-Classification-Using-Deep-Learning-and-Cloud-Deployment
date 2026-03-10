# 🛣️ Automated Road Damage Classification Using Deep Learning and Cloud Deployment

A Deep Learning and Computer Vision project designed to automatically identify and classify road infrastructure issues (**Potholes**, **Cracks**, and **Manholes**) and provide actionable maintenance recommendations using **Explainable AI (Grad-CAM)**.

---

## 🚀 Overview

Road infrastructure maintenance is critical for public safety. Manual inspection is time-consuming, inconsistent, and costly. This project provides an **AI-powered solution** using **Convolutional Neural Networks (CNN)** and **Transfer Learning** to process road images captured from smartphones or vehicle cameras, delivering real-time analysis for municipal authorities and transportation departments.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔍 **Multi-Class Classification** | Detects 3 damage categories: Potholes, Cracks, and Manholes |
| 🧠 **Explainable AI (XAI)** | Uses **Grad-CAM** heatmaps to show which image regions influenced the AI's decision |
| 🛠️ **Maintenance Recommendations** | Provides damage-specific repair actions for prioritization |
| ⚡ **Transfer Learning** | Fine-tuned **ResNet50** for robust performance on limited data |
| 🌐 **Web Application** | Built with **Streamlit** for an interactive, user-friendly experience |
| 📊 **Model Comparison** | Evaluated Baseline CNN, MobileNetV2, and ResNet50 architectures |

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python |
| **Deep Learning** | TensorFlow, Keras |
| **Image Processing** | OpenCV, PIL, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Explainability** | Grad-CAM (Gradient-weighted Class Activation Mapping) |
| **Data Handling** | Pandas, Scikit-learn |
| **Model Architectures** | Baseline CNN, MobileNetV2, ResNet50 |
| **Deployment** | Streamlit (Local / Cloud) |

---

## 📊 Approach & Methodology

### 1️⃣ Data Preparation
- **Dataset**: Road Damage Dataset (RDD) with 3 classes — Pothole, Crack, Manhole
- **Preprocessing**: Resized to `224×224`, pixel normalization (`/255.0`)
- **Data Augmentation**: Rotation, horizontal/vertical flips, brightness adjustment, blur for real-world robustness
- **Class Imbalance**: Handled using `class_weights` to prevent bias towards majority class
- **Split**: Train / Test split for model evaluation

### 2️⃣ Model Development

| Model | Type | Description |
|---|---|---|
| **Baseline CNN** | Custom CNN | Built from scratch with Conv2D, MaxPool, Dense layers |
| **MobileNetV2** | Transfer Learning | Lightweight model, fine-tuned last 50 layers + BatchNorm |
| **ResNet50** | Transfer Learning | Deep residual network, fine-tuned for best accuracy ✅ |

#### Fine-Tuning Strategy (MobileNetV2 / ResNet50):
```python
# Step 1: Unfreeze base model
base_model.trainable = True

# Step 2: Freeze first layers (keep early feature extraction frozen)
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Step 3: Re-enable BatchNormalization layers for domain adaptation
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True
```

### 3️⃣ Model Evaluation

| Model | Accuracy | Status |
|---|---|---|
| Baseline CNN | ~51% | ❌ Weak |
| MobileNetV2 (Fine-Tuned) | ~54% | ⚠️ Moderate |
| **ResNet50 (Fine-Tuned)** | **~61%** | ✅ **Best Model** 🏆 |

- **ResNet50** was selected as the final model due to its superior feature extraction and highest F1-score
- Evaluation includes: Confusion Matrix, Classification Report, and Grad-CAM visualizations

### 4️⃣ Explainable AI — Grad-CAM
- Generates **heatmaps** overlaid on original images
- Shows exactly **which regions** of the road image the model focused on
- Helps build **trust and transparency** in AI predictions

---

## 📂 Project Structure

```text
Final Project/
├── data/
│   ├── images/                          # Dataset image files
│   ├── labels/                          # OBB/Text annotations
│   ├── train/                           # Training data
│   └── test/                            # Test data
├── TEST.ipynb                           # Training & Evaluation Notebook
├── app.py                               # Streamlit Web Application
├── test_gradcam.py                      # Grad-CAM Testing Script
├── baseline_cnn.h5                      # Baseline CNN Model
├── mobilenetv2_road_damage.h5           # MobileNetV2 Fine-Tuned Model
├── resnet50_fine_tuned.h5               # ResNet50 Fine-Tuned Model (Best) 🏆
├── Final Project_B94.pdf                # Project Report
└── README.md                            # Documentation
```

---

## ⚙️ How to Run

### Prerequisites
```bash
pip install tensorflow streamlit numpy pillow opencv-python matplotlib seaborn scikit-learn pandas
```

### 1. Training the Model
Open the Jupyter Notebook and run all cells:
```bash
jupyter notebook TEST.ipynb
```
> This will train all 3 models and generate `.h5` model files.

### 2. Launching the Web App
Run the following command in your terminal:
```bash
streamlit run app.py
```
> The app will open at `http://localhost:8501` — upload a road image to get predictions!

---

## 🌐 Web App Features

1. **Upload** a road image (JPG, JPEG, PNG)
2. **AI Prediction** — Detects damage type (Pothole / Crack / Manhole)
3. **Confidence Score** — Shows prediction confidence percentage
4. **Class Probabilities** — Displays probability for all 3 classes
5. **Maintenance Recommendation** — Provides actionable repair suggestions

| Damage Type | Recommendation |
|---|---|
| ⚠️ Pothole | Schedule immediate repair |
| 🔍 Crack | Monitor and seal the crack soon |
| ✅ Manhole | Inspect cover condition and alignment |

---

## 📈 Evaluation Metrics Summary

The final evaluated performance across all tested architectures:

| Metric | Baseline CNN | MobileNetV2 FT | ResNet50 FT 🏆 |
|---|---|---|---|
| **Accuracy** | ~51% | ~54% | **~61%** |
| **Best For** | Baseline comparison | Lightweight deployment | Best accuracy |
| **Layers Fine-Tuned** | All (custom) | Last 50 + BatchNorm | Last layers + BatchNorm |

> **Conclusion**: ResNet50 achieved the highest accuracy and F1-score, demonstrating superior feature extraction capabilities for road damage detection compared to lighter networks.

---

## 🔮 Future Improvements

- 📷 **Real-time Detection**: Integrate with live camera feeds for continuous road monitoring
- 🗺️ **GPS Integration**: Map damaged locations for city-wide damage tracking
- 📱 **Mobile App**: Deploy on Android/iOS for on-field inspections
- 🧠 **More Classes**: Extend to detect speed bumps, lane markings, and road signs
- ☁️ **Cloud Deployment**: Deploy on AWS/GCP/Azure for scalable usage

---

## 👨‍💻 Author

**Hariharan** — Data Science Intern  
Project: Automated Road Damage Classification Using Deep Learning and Cloud Deployment

---

## 📝 License

This project is developed for educational and research purposes.

---

> *Built with ❤️ using TensorFlow, ResNet50, and Streamlit*
