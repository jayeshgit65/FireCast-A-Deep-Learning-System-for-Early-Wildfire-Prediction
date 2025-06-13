import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load models
cnn_model = load_model(r"C:\Users\lenovo\Documents\Projects\Wildlife_Prediction\wildfire_cnn_best_model.h5")
resnet_model = load_model(r"C:\Users\lenovo\Documents\Projects\Wildlife_Prediction\wildfire_resnet50_best_model.h5")
class_names = ['No Wildfire', 'Wildfire']

def preprocess_image(image_path, size=(128, 128)):
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.resize(image, size)
        image = image / 255.0  # Normalize
        return np.expand_dims(image, axis=0)
    return None

# Streamlit UI
st.title("Wildfire Image Classifier")
st.write("Upload an image to classify it as **Wildfire** or **No Wildfire**.")

model_choice = st.radio("Choose Model", ["Custom CNN", "ResNet50"])

uploaded_file = st.file_uploader("Upload satellite/fire image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    temp_path = "temp_fire_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(Image.open(temp_path), caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    preprocessed = preprocess_image(temp_path)
    if preprocessed is not None:
        model = cnn_model if model_choice == "Custom CNN" else resnet_model
        pred = model.predict(preprocessed)
        label = 1 if pred >= 0.5 else 0
        confidence = float(pred[0][0]) * 100 if label == 1 else (1 - float(pred[0][0])) * 100

        st.success(f"**Prediction:** {class_names[label]}")
        st.info(f"**Confidence:** {confidence:.2f}%")
    else:
        st.error("Failed to read or preprocess the image.")
