import streamlit as st
import cv2
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
from datetime import datetime
import os
import time

# Constants
MODEL_NAME = "facebook/dino-vits16"
LABELS = ["Not Drowsy", "Drowsy"]
THRESHOLD = 0.6  # Adjust based on testing
USER_CREDENTIALS = {"user": "123"}
ADMIN_CREDENTIALS = {"admin": "admin123"}

# Store session predictions
if "predictions" not in st.session_state:
    st.session_state["predictions"] = []

# Ensure misclassified images are stored for debugging
MISCLASSIFIED_DIR = "misclassified_images"
os.makedirs(MISCLASSIFIED_DIR, exist_ok=True)

def authenticate(username, password, role):
    if role == "User" and username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
        return True
    elif role == "Admin" and username in ADMIN_CREDENTIALS and ADMIN_CREDENTIALS[username] == password:
        return True
    return False

@st.cache_resource
def load_model():
    """ Load the image classification model and feature extractor """
    try:
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        return model, feature_extractor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_image(image, feature_extractor):
    """ Convert and preprocess image for model input """
    image = image.convert("RGB")
    image = image.resize((224, 224))  # Ensure correct input size
    return feature_extractor(images=image, return_tensors="pt")

def get_prediction(model, inputs):
    """ Get model prediction and confidence score """
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)  # Normalize scores
            predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
            prediction_score = probabilities[0, predicted_class_idx].item()

        # Adjust prediction using threshold
        if prediction_score < THRESHOLD:
            predicted_class_idx = 1  # Force "Drowsy"

        return predicted_class_idx, prediction_score
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def display_result(image, predicted_class_idx, prediction_score):
    """ Display prediction result with confidence score """
    st.image(image, caption="Captured Image", use_column_width=True)
    if predicted_class_idx is not None:
        prediction_label = LABELS[predicted_class_idx]
        st.write(f"**Prediction:** {prediction_label}  \n"
                 f"**Confidence Score:** {prediction_score:.2f}")

        # Save prediction with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["predictions"].append({
            "Prediction": prediction_label, 
            "Confidence Score": f"{prediction_score:.2f}",
            "Timestamp": timestamp
        })

        # Save misclassified images for debugging
        if prediction_label == "Not Drowsy" and prediction_score < THRESHOLD:
            image_path = os.path.join(MISCLASSIFIED_DIR, f"{timestamp.replace(':', '-')}.jpg")
            image.save(image_path)
            st.write(f"⚠️ Misclassified image saved to: `{image_path}`")

def sidebar():
    """ Sidebar authentication for users and admins """
    st.sidebar.title("Drowsiness Detection System")
    role = st.sidebar.radio("Select Role", ("User", "Admin"))
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if authenticate(username, password, role):
            st.session_state["authenticated"] = True
            st.session_state["role"] = role
            st.sidebar.success(f"Logged in as {role}")
        else:
            st.sidebar.error("Invalid credentials. Please try again.")

def live_detection(model, feature_extractor):
    """ Live Webcam-Based Drowsiness Detection """
    st.subheader("Live Detection Mode")
    st.markdown("### 🚗 Detecting drowsiness in real time.")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Cannot access webcam. Make sure it's not being used by another application.")
        return

    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to capture frame.")
            break

        # Convert to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # Preprocess and predict
        inputs = preprocess_image(pil_img, feature_extractor)
        predicted_class_idx, prediction_score = get_prediction(model, inputs)
        
        # Display results
        label = LABELS[predicted_class_idx]
        confidence = f"{prediction_score:.2f}"

        # Draw result on frame
        cv2.putText(frame, f"Prediction: {label} ({confidence})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        stframe.image(frame, channels="BGR", use_column_width=True)

        # Stop detection
        if st.button("Stop Detection"):
            break

        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

def main():
    """ Main app logic """
    st.title("Real-Time Drowsiness Detection")
    st.markdown("This application detects drowsiness using a deep learning model.")
    sidebar()
    
    if "authenticated" not in st.session_state:
        return
    
    role = st.session_state.get("role", "User")
    
    if role == "User":
        model, feature_extractor = load_model()
        if model is None or feature_extractor is None:
            st.error("Failed to load the model. Please check your internet connection or try again later.")
            return

        # Live Webcam Detection
        if st.button("Start Live Detection"):
            live_detection(model, feature_extractor)

    else:  # Admin Panel
        st.title("Admin Dashboard")
        st.write("Below are the recorded predictions with date and time:")
        
        if st.session_state["predictions"]:
            df = pd.DataFrame(st.session_state["predictions"])
            st.dataframe(df)
        else:
            st.write("No data available.")

if __name__ == "__main__":
    main()
