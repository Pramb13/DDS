import streamlit as st
import cv2
import numpy as np
import requests
from PIL import Image
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

# Model setup
MODEL_NAME = "facebook/dino-vits16"
LABELS = ["Not Drowsy", "Drowsy"]
THRESHOLD = 0.6

@st.cache_resource
def load_model():
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    return model, feature_extractor

def preprocess_image(image, feature_extractor):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    return feature_extractor(images=image, return_tensors="pt")

def get_prediction(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
        prediction_score = probabilities[0, predicted_class_idx].item()
    return predicted_class_idx, prediction_score

# Streamlit UI
st.title("Live Drowsiness Detection")

video_url = st.text_input("Enter webcam stream URL", "http://your_local_server/video_feed")
if st.button("Start Live Detection"):
    model, feature_extractor = load_model()
    
    if not video_url.startswith("http"):
        st.error("Invalid video URL. Provide a valid streaming link.")
    else:
        while True:
            try:
                response = requests.get(video_url, stream=True, timeout=5)
                bytes_data = bytes()
                for chunk in response.iter_content(chunk_size=1024):
                    bytes_data += chunk
                    a = bytes_data.find(b'\xff\xd8')
                    b = bytes_data.find(b'\xff\xd9')
                    if a != -1 and b != -1:
                        jpg = bytes_data[a:b+2]
                        bytes_data = bytes_data[b+2:]
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(frame)

                        # Run prediction
                        inputs = preprocess_image(image, feature_extractor)
                        predicted_class_idx, prediction_score = get_prediction(model, inputs)
                        prediction_label = LABELS[predicted_class_idx]

                        # Display
                        st.image(image, caption=f"Prediction: {prediction_label} ({prediction_score:.2f})", use_column_width=True)
                        break  # Refresh frame
            except Exception as e:
                st.error(f"Failed to fetch video: {e}")
                break
