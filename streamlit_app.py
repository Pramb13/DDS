import streamlit as st
import torch
import cv2
import numpy as np
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import time

# Constants
MODEL_NAME = "facebook/dino-vits16"
LABELS = ["Not Drowsy", "Drowsy"]
THRESHOLD = 0.6

# Load Model
@st.cache_resource
def load_model():
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    return model, feature_extractor

def preprocess_image(image, feature_extractor):
    """ Convert and preprocess image for model input """
    image = Image.fromarray(image)  # Convert from OpenCV to PIL
    image = image.resize((224, 224))  # Resize to model input size
    return feature_extractor(images=image, return_tensors="pt")

def get_prediction(model, inputs):
    """ Get model prediction and confidence score """
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
        prediction_score = probabilities[0, predicted_class_idx].item()

    # Force classification as "Drowsy" if below threshold
    if prediction_score < THRESHOLD:
        predicted_class_idx = 1

    return predicted_class_idx, prediction_score

def main():
    """ Main app logic """
    st.title("Live Drowsiness Detection")
    st.markdown("### 🚗 Detecting drowsiness in real time using your webcam.")

    model, feature_extractor = load_model()
    if model is None or feature_extractor is None:
        st.error("❌ Failed to load the model.")
        return

    # Start/Stop Button
    start_button = st.button("Start Detection")
    stop_button = st.button("Stop Detection")

    # Create an empty container to display frames
    stframe = st.empty()

    if start_button:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot access webcam.")
            return

        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to capture frame.")
                break

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Preprocess and Predict
            inputs = preprocess_image(frame_rgb, feature_extractor)
            predicted_class_idx, prediction_score = get_prediction(model, inputs)
            prediction_label = LABELS[predicted_class_idx]

            # Draw label on the frame
            text = f"{prediction_label} ({prediction_score:.2f})"
            color = (0, 255, 0) if predicted_class_idx == 0 else (0, 0, 255)
            cv2.putText(frame_rgb, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # Show frame in Streamlit
            stframe.image(frame_rgb, channels="RGB", use_column_width=True)

            # Small delay to avoid overloading CPU
            time.sleep(0.1)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
