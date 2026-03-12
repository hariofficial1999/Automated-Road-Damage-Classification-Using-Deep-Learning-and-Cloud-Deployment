# 🚧 Automated Road Damage Classification

## 📌 Project Overview
Road maintenance is critical for public safety and vehicle longevity. This project leverages **Deep Learning and Computer Vision** to automatically detect and classify road anomalies. Using **Transfer Learning** with state-of-the-art architectures, the system identifies three primary types of road conditions:
- **Potholes** 🕳️
- **Cracks** 🔍
- **Manholes** 🔘

The final solution includes a trained model and a **Streamlit-based web application** for real-time inference.

---

## 🚀 Features
- **Multi-Class Classification**: Identifies Potholes, Cracks, and Manholes.
- **High Performance**: Utilizes fine-tuned **ResNet50** and **MobileNetV2** architectures.
- **Actionable Insights**: Provides maintenance recommendations based on detection results.
- **Interactive UI**: User-friendly web interface for image uploads and instant classification.
- **Full Pipeline**: Includes data preprocessing, model training, evaluation, and deployment.

---

## 📊 Model Performance
We evaluated multiple models to find the best-performing architecture. Below is a comparison of the key metrics obtained during the testing phase:

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
| :--- | :---: | :---: | :---: | :---: |
| **ResNet50 FT** | **0.4279** | **0.4848** | **0.4541** | **0.4170** |
| MobileNetV2 FT | 0.4234 | 0.4736 | 0.4071 | 0.3715 |
| Baseline CNN | 0.3018 | 0.4384 | 0.3756 | 0.2617 |

*Note: Performance reflects the complexity of real-world road imagery and can be further improved with larger datasets and advanced augmentation techniques.*

---

## 🛠️ Technology Stack
- **Languages**: Python
- **Deep Learning**: TensorFlow / Keras
- **Web Framework**: Streamlit
- **Data Handling**: NumPy, Pandas, PIL
- **Visualization**: Matplotlib, Seaborn

---

## 📁 Project Structure
```text
├── data/               # Raw images and labels
├── split/              # Processed train/test/val dataset
├── app.py              # Streamlit Web Application
├── Test.ipynb          # Training & Evaluation Notebook
├── resnet50_fine_tuned.h5  # Best Saved Model
├── baseline_cnn.h5     # Baseline Model
└── mobilenetv2_road_damage.h5 # MobileNetV2 Model
```

---

## 💻 Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/HARIHARAN-Data-Scientist/Automated-Road-Damage-Classification-Using-Deep-Learning-and-Cloud-Deployment.git
cd Automated-Road-Damage-Classification-Using-Deep-Learning-and-Cloud-Deployment
```

### 2. Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install tensorflow streamlit pillow numpy
```

### 3. Run the Web App
```bash
streamlit run app.py
```

---

## 🖼️ User Interface
The Streamlit app allows users to:
1. Upload a road image.
2. View the classification result and confidence score.
3. Receive immediate maintenance suggestions (e.g., "Schedule immediate repair" for potholes).

---

## 📝 Future Improvements
- [ ] Integration with GPS for geotagged damage reporting.
- [ ] Implementation of YOLOv8 for object detection (bounding boxes).
- [ ] Expansion of dataset to include more weather conditions and road types.

---

**Developed by [HARIHARAN-Data-Scientist](https://github.com/HARIHARAN-Data-Scientist)**
