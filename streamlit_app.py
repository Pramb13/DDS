import streamlit as st
import torch
import numpy as np
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import os

# Load Model
MODEL_NAME = "facebook/dino-vits16"
LABELS = ["Not Drowsy", "Drowsy"]
THRESHOLD = 0.6  # Adjust based on testing

@st.cache_resource
def load_model():
    """ Load the image classification model and feature extractor """
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    return model, feature_extractor

def preprocess_image(image, feature_extractor):
    """ Preprocess image for model input """
    image = image.convert("RGB")
    image = image.resize((224, 224))  # Resize to match model input
    return feature_extractor(images=image, return_tensors="pt")

def get_prediction(model, inputs):
    """ Perform inference and return result """
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
        prediction_score = probabilities[0, predicted_class_idx].item()

    # Adjust prediction using threshold
    if prediction_score < THRESHOLD:
        predicted_class_idx = 1  # Force "Drowsy"

    return LABELS[predicted_class_idx], prediction_score

def main():
    """ Streamlit UI """
    st.title("🚘 Driver Drowsiness Detection")
    st.markdown("Upload an image to detect drowsiness.")

    # Load model
    model, feature_extractor = load_model()

    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict
        inputs = preprocess_image(image, feature_extractor)
        label, score = get_prediction(model, inputs)

        # Display Result
        st.markdown(f"### **Prediction: {label}**")
        st.markdown(f"**Confidence Score: {score:.2f}**")

if __name__ == "__main__":
    main()
