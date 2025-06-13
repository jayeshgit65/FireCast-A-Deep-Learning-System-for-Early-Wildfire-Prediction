# Wildfire Prediction Analysis

This project is a user-friendly web application for wildfire prediction analysis which classifies images as **Wildfire** or **No Wildfire** using deep learning models. In this research, the custom Convolutional Neural Network (CNN) outperformed 
the pretrained ResNet50 model, achieving an accuracy of 94% compared to 86%. The app allows users to upload satellite or fire images and choose between two models: a custom CNN or a pretrained ResNet50 model. 

---

## Features

- Upload images directly without saving them locally.
- Supports JPG, JPEG, and PNG image formats.
- Choose between a custom Convolutional Neural Network (CNN) and pretrained ResNet50 model for classification.
- Displays prediction results along with confidence scores.
- Clean and intuitive user interface built with Streamlit.
  
---

## Installation

1. Make sure you have Python 3.7 or higher installed.

2. Clone the repository or download the project files.

3. Install the required Python packages:
   ```bash
   pip install streamlit tensorflow pillow opencv-python-headless numpy
