import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

# Load the trained model
model = keras.models.load_model("C:\\Users\\a\\Desktop\\ImageCaptioning\digit_recognition_model.h5")

# Streamlit app
st.title("Digit Recognition App")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Preprocess the image
    img = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to the input size of the model
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = img_array.reshape(1, 784)  # Flatten to match the model's input shape

    # Display the uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.write(f"Prediction: {predicted_class}")
