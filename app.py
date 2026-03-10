import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Page settings
st.set_page_config(page_title="Road Damage Classification", layout="centered")
st.title("🚧 Road Damage Classification ")
st.write("Upload a road image to classify the damage type (Pothole, Crack, Manhole).")

# Model path
MODEL_PATH = "resnet50_fine_tuned.h5"

# Load trained model
@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Cannot find model file: {MODEL_PATH}")
        return None
    # Supply preprocess_input as a custom object so TensorFlow can successfully deserialize the Lambda layer
    model = tf.keras.models.load_model(MODEL_PATH, compile=False, custom_objects={'preprocess_input': tf.keras.applications.resnet50.preprocess_input})
    # Warmup prediction
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    _ = model(dummy_input, training=False)
    return model

model = load_my_model()

if model is None:
    st.stop()
    
# Class names & Recommendations
classes = ["Pothole", "Crack", "Manhole"]
recommendations = {
    "Pothole": "⚠️ Recommended Action: Schedule immediate repair.",
    "Crack": "🔍 Recommended Action: Monitor and seal the crack soon.",
    "Manhole": "✅ Recommended Action: Inspect manhole cover condition and alignment."
}

# Image preprocessing
def preprocess_image(image):
    # Resize to exactly 224x224 as required by the model
    img = image.resize((224, 224))
    img_array = np.array(img).astype(np.float32)
    
    # Handle RGBA images
    if img_array.ndim == 3 and img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
        
    # Handle Grayscale images
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
        
    # Add batch dimension: shape becomes (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 🚨 NOT applying preprocess_input here because the model built
    # in Jupyter ALREADY has a Lambda(preprocess_input) layer! 
    # Applying it twice would ruin the predictions.
    return img_array

# File uploader & Prediction
uploaded_file = st.file_uploader("Upload an image (Pothole, Crack, Manhole)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    # Predict
    img_array = preprocess_image(image)
    prediction = model.predict(img_array, verbose=0)
    pred_index = int(np.argmax(prediction[0]))
    pred_class = classes[pred_index]
    confidence = float(np.max(prediction[0]) * 100)

    st.subheader("Prediction Result")
    st.write(f"**Detected Damage:** {pred_class}")
    st.write(f"**Confidence Score:** {confidence:.2f}%")

    st.subheader("Recommendation")
    st.info(recommendations[pred_class])

    st.subheader("Class Probabilities")
    for i, class_name in enumerate(classes):
        st.write(f"**{class_name}:** {prediction[0][i] * 100:.2f}%")

st.markdown("---")
st.write("Built with Streamlit and ResNet50 Transfer Learning")
