import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Load models
cnn_model = load_model(os.path.join("models", "wildfire_cnn_best_model.h5"))
resnet_model = load_model(os.path.join("models", "wildfire_resnet50_best_model.h5"))

class_names = ['No Wildfire', 'Wildfire']

def preprocess_image(image, size=(128, 128)):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert PIL image to OpenCV format
    image = cv2.resize(image, size)
    image = image / 255.0  # Normalize
    return np.expand_dims(image, axis=0)

# Streamlit UI
st.set_page_config(page_title="Wildfire Image Classifier", layout="centered")
st.title("🌲 Wildfire Predication")
st.write("Upload an image to predict **wildfire**")

model_choice = st.radio("Choose Model", ["Custom CNN", "ResNet50"], index=0)

uploaded_file = st.file_uploader("Upload satellite/fire image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Preprocess the image
    preprocessed = preprocess_image(image)
    if preprocessed is not None:
        model = cnn_model if model_choice == "Custom CNN" else resnet_model
        pred = model.predict(preprocessed)
        label = 1 if pred >= 0.5 else 0
        confidence = float(pred[0][0]) * 100 if label == 1 else (1 - float(pred[0][0])) * 100

        # Display the prediction results
        st.success(f"**Prediction:** {class_names[label]}")
        st.info(f"**Confidence:** {confidence:.2f}%")
    else:
        st.error("Failed to preprocess the image.")
else:
    st.warning("Please upload an image to get a prediction.")
