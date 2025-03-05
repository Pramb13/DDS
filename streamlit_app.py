import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image

# Load Haar Cascade for Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load Pretrained Model
@st.cache_resource
def load_model():
    model_name = "facebook/dino-vits16"
    model = AutoModelForImageClassification.from_pretrained(model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    return model, feature_extractor

model, feature_extractor = load_model()

# Preprocess Image for Model
def preprocess_image(image):
    image = Image.fromarray(image)
    image = image.convert("RGB").resize((224, 224))
    return feature_extractor(images=image, return_tensors="pt")

# Get Prediction
def get_prediction(image):
    inputs = preprocess_image(image)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
        confidence_score = probabilities[0, predicted_class_idx].item()
    return predicted_class_idx, confidence_score

# Video Processing Class
class VideoProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        # Detect Faces
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face = frm[y:y+h, x:x+w]
            if face.size > 0:
                label, confidence = get_prediction(face)
                status = "Drowsy" if label == 1 else "Not Drowsy"
                
                # Draw Rectangle & Label
                color = (0, 0, 255) if status == "Drowsy" else (0, 255, 0)
                cv2.rectangle(frm, (x, y), (x+w, y+h), color, 3)
                cv2.putText(frm, f"{status} ({confidence:.2f})", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# Streamlit UI
st.title("🚗 Driver Drowsiness Detection")
st.write("This application detects drowsiness in real-time.")

# Live Video Stream
webrtc_streamer(
    key="drowsiness_detection",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
)

