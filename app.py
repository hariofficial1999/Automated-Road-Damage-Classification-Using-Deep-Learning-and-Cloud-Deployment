#Import libraries
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Page settings
st.set_page_config(page_title="Road Damage Classification", layout="centered")

st.title("Road Damage Classification App")
st.write("Upload a road image to classify the damage type.")

# Model loading
MODEL_PATH = "resnet50_fine_tuned.h5"

# Load trained model
@st.cache_resource
def load_my_model():
    # Load the base model
    base_model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": preprocess_input},
        compile=False
    )
    
    # Rebuild as a functional model to ensure model.input and model.output are defined safely
    inputs = tf.keras.Input(shape=(224, 224, 3), name="app_input_layer")
    x = inputs
    for layer in base_model.layers:
        x = layer(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    
    # Call the model once using dummy input after loading
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    _ = model(dummy_input)
    
    return model


# Load model with error handling
try:
    model = load_my_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Model load aagala. Check model path.\n\nError: {e}")
    st.stop()

# Class names
classes = ["Pothole", "Crack", "Manhole"]

# Recommendation messages
recommendations = {
    "Pothole": "Recommended Action: Schedule immediate repair.",
    "Crack": "Recommended Action: Monitor and seal the crack soon.",
    "Manhole": "Recommended Action: Inspect manhole cover condition and alignment."}

# Image preprocessing function
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)

    # RGB ensure panrom
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array.astype(np.float32))
    
    return img_array

# Grad-CAM function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    # automatic-ah last conv layer find panna
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            try:
                if len(layer.output.shape) == 4:
                    last_conv_layer_name = layer.name
                    break
            except:
                continue

    # Build Grad-CAM safely using model layers dynamically to avoid Keras 3 Functional.call() KeyError
    grad_model_inputs = tf.keras.Input(shape=(224, 224, 3))
    x = grad_model_inputs
    conv_output = None
    
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        x = layer(x)
        if layer.name == last_conv_layer_name:
            conv_output = x
            
    grad_model = tf.keras.models.Model(
        grad_model_inputs,
        [conv_output, x]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

# Heatmap overlay function

def overlay_heatmap(original_image, heatmap, alpha=0.4):
    original = np.array(original_image)
    
    if original.shape[-1] == 4:
        original = original[:, :, :3]

    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original, 1 - alpha, heatmap_colored, alpha, 0)

    return superimposed_img

# File uploader

uploaded_file = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])

# Prediction block

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    st.subheader("Uploaded Image")
    st.image(image, caption="Uploaded Road Image", use_container_width=True)

    img_array = preprocess_image(image)

    prediction = model.predict(img_array)
    pred_index = np.argmax(prediction[0])
    pred_class = classes[pred_index]
    confidence = float(np.max(prediction[0]) * 100)

    st.subheader("Prediction Result")
    st.write(f"**Detected Damage:** {pred_class}")
    st.write(f"**Confidence Score:** {confidence:.2f}%")

    st.subheader("Recommendation")
    st.info(recommendations[pred_class])

# Show class probabilities
    st.subheader("Class Probabilities")
    for i, class_name in enumerate(classes):
        st.write(f"**{class_name}:** {prediction[0][i] * 100:.2f}%")

# Grad-CAM display
    st.subheader("Grad-CAM Heatmap")

    try:
        heatmap = make_gradcam_heatmap(img_array, model)
        gradcam_result = overlay_heatmap(image, heatmap)

        st.image(gradcam_result, caption="Grad-CAM Visualization", use_container_width=True)
    except Exception as e:
        st.warning(f"Grad-CAM generate panna mudiyala.\nError: {e}")

# Footer
st.markdown("---")
st.write("Built with Streamlit + ResNet50 for Road Damage Classification")