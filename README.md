# Automated-Road-Damage-Classification-Using-Deep-Learning-and-Cloud-Deployment

# 🛣️ Automated Road Damage Classification System

A Deep Learning and Computer Vision project designed to automatically identify and classify road infrastructure issues (Potholes, Cracks, and Manholes) and provide actionable maintenance recommendations.

---

## 🚀 Overview
Road infrastructure maintenance is critical for safety. This project provides an AI-powered solution using **Convolutional Neural Networks (CNN)** and **Transfer Learning** to process road images captured from smartphones or vehicle cameras, providing real-time analysis for municipal authorities and transportation departments.

## ✨ Key Features
- **Classification**: Detects 3 categories: Potholes, Cracks, and Manholes.
- **Explainable AI (XAI)**: Uses **Grad-CAM** to generate heatmaps, showing exactly which regions of the image influenced the AI's decision.
- **Maintenance Recommendations**: Provides damage-specific actions for repair prioritization.
- **Lightweight Deployment**: Optimized using **ResNet50** for robust performance on a web interface.
- **Modern UI**: Built with **Streamlit** for a user-friendly experience.

---

## 🛠️ Tech Stack
- **Languages**: Python
- **Libraries**: TensorFlow, Keras, OpenCV, NumPy, Matplotlib, Seaborn
- **Architecture**: ResNet50 (Best Model / Transfer Learning), Baseline CNN
- **Deployment**: Streamlit Local / Cloud

---

## 📊 Approach
1.  **Data Preparation**:
    - Dataset: Road Damage Dataset (RDD)
    - Preprocessing: 224x224 resizing, normalization.
    - Augmentation: Rotation, flips, brightness adjustment, and blur for real-world robustness.
    - Class Imbalance: Handled using `class_weights`.
2.  **Model Development**:
    - Baseline CNN model.
    - Transfer Learning with **ResNet50** (Selected as the Best Model).
    - Fine-tuning selected layers to optimize accuracy.
3.  **Model Evaluation**:
    - Compared 3 architectures: Baseline CNN (51%), MobileNetV2 FT (54%), and **ResNet50 FT (61%)**.
    - **ResNet50** was selected as the final best model due to its superior feature extraction and highest F1-score.
    - Visualizations: Confusion Matrix and Grad-CAM screenshots.

---

## 📂 Project Structure
```text
D:\Intern Project\Final Project\
├── data/
│   ├── images/      # Dataset image files
│   └── labels/      # OBB/Text annotations
├── Road_Damage_Classification.ipynb  # Training & Evaluation logic
├── app.py           # Streamlit Web Application
├── resnet50_fine_tuned.h5       # Trained model file (Best Model)
└── README.md        # Documentation
```

---

## ⚙️ How to Run

### 1. Training the Model
Open the Jupyter Notebook and run all cells:
```bash
jupyter notebook Road_Damage_Classification.ipynb
```
*This will generate the `resnet50_fine_tuned.h5` file.*

### 2. Launching the Web App
Run the following command in your terminal:
```bash
streamlit run app.py
```

---

## 📈 Evaluation Metrics
The final evaluated performance across tested architectures:
- **Baseline CNN**: ~51% Accuracy
- **MobileNetV2 (Transfer Learning)**: ~54% Accuracy
- **ResNet50 (Transfer Learning)**: ~**61% Accuracy** 🏆 *(Best Model)*

ResNet50 achieved the highest accuracy and F1-score, demonstrating superior feature extraction capabilities for road damage detection compared to lighter networks.
- **System Metrics**: Latency analysis for real-time responsiveness.
- **Usability**: Ease of use for manual damage reporting.
