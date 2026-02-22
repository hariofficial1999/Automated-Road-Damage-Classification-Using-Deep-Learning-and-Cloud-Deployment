import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# Page Config
st.set_page_config(page_title="Road Damage Classifier", layout="wide")

# LOAD MODEL
@st.cache_resource
def load_trained_model():
    # If the model isn't trained yet, we'll show an error. 
    model_path = "road_damage_final_model.h5"
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        # Force the model to initialize its functional state by running a dummy input
        # This is more robust than model.build() in Keras 3
        dummy_input = tf.ones((1, 224, 224, 3))
        _ = model(dummy_input)
        return model
    return None

model = load_trained_model()

# CONSTANTS
BASE_DIR = r"D:\Intern Project\Final Project\data"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
CLASSES = ['Pothole', 'Crack', 'Manhole']
RECOMMENDATIONS = {
    'Pothole': "Detected: Pothole → Recommended Action: Schedule immediate repair to prevent vehicle damage.",
    'Crack': "Detected: Crack → Recommended Action: Fill cracks to prevent water penetration and potholes.",
    'Manhole': "Detected: Manhole → Recommended Action: Inspect manhole cover for unevenness or structural weakness."
}

# Grad-CAM Function for Web-App
def get_gradcam_overlay(img_array, model):
    img_tensor = tf.cast(img_array, tf.float32)
    
    # Extract the base model (MobileNetV2) - Layer 0 of our Sequential app
    base_model = model.layers[0]
    
    # Find the last convolutional layer
    last_conv_layer = None
    for layer in reversed(base_model.layers):
        if 'conv' in layer.name or 'project' in layer.name:
            last_conv_layer = layer
            break
            
    if not last_conv_layer:
        return None

    # Step-by-step propagation within the GradientTape to ensure gradient connectivity
    with tf.GradientTape() as tape:
        # 1. Get the last convolutional output AND the base model output
        # We create a sub-model for this part of the computation
        base_grad_model = tf.keras.Model(base_model.input, [last_conv_layer.output, base_model.output])
        conv_output, base_out = base_grad_model(img_tensor)
        
        # 2. Pass the base_model's output through the REMAINING layers of the Sequential model
        x = base_out
        for i in range(1, len(model.layers)):
            x = model.layers[i](x)
        
        predictions = x
        top_pred_index = tf.argmax(predictions[0])
        top_score = predictions[:, top_pred_index]

    # Compute the gradient of the top score with respect to the convolutional feature map
    grads = tape.gradient(top_score, conv_output)

    if grads is None:
        return None

    # Pool the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Calculate heatmap from the weighted combination of feature map channels
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # Process for display (Resize to original and colorize)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    return heatmap

# UI DESIGN
st.title("🛣️ Automated Road Damage Classification")
st.markdown("""
### Deep Learning System for Smart Cities
Select an image from the dataset or upload your own to identify Potholes, Cracks, or Manhole issues.
""")

# --- IMAGE SELECTION SIDEBAR ---
st.sidebar.title("Dataset Explorer")
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
image_files.sort()

selected_img_name = st.sidebar.selectbox("Select Image from Dataset", ["Upload New..."] + image_files)
uploaded_file = st.file_uploader("Choose a road image...", type=["jpg", "jpeg", "png"])

# Logic to decide which image to use
target_image = None

if uploaded_file is not None:
    target_image = Image.open(uploaded_file).convert('RGB')
elif selected_img_name != "Upload New...":
    target_path = os.path.join(IMAGE_DIR, selected_img_name)
    target_image = Image.open(target_path).convert('RGB')

if target_image is not None:
    # 1. Preprocess
    display_img = np.array(target_image)
    img = target_image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    if model is not None:
        # 2. Predict
        preds = model.predict(img_array)
        class_idx = np.argmax(preds[0])
        label = CLASSES[class_idx]
        confidence = preds[0][class_idx] * 100

        col1, col2 = st.columns(2)
        with col1:
            st.image(target_image, caption="Current Image", use_container_width=True)
            st.metric("Detected Category", label)
            st.write(f"**Confidence Score:** {confidence:.2f}%")
            st.success(RECOMMENDATIONS[label])

        with col2:
            # 3. Grad-CAM Visualization
            st.write("### Explainable AI (Grad-CAM)")
            heatmap = get_gradcam_overlay(img_array, model)
            
            # Superimpose
            original_opencv = cv2.resize(np.array(target_image), (224, 224))
            overlayed_img = cv2.addWeighted(original_opencv, 0.6, heatmap, 0.4, 0)
            
            st.image(overlayed_img, caption="Grad-CAM Heatmap (Damage regions)", use_container_width=True)
            st.info("The highlighted areas (Red/Yellow) indicate where the AI 'looked' to make its decision.")
    else:
        st.warning("⚠️ Model file 'road_damage_final_model.h5' not found. Please run the notebook first to train the model.")
