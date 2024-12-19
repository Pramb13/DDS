import streamlit as st
import cv2
import torch
from PIL import Image
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

# Load YOLOv5 model
@st.cache_resource
def load_model():
    device = select_device("")
    model = DetectMultiBackend("best.pt", device=device)
    return model

# Process video frames
def detect_drowsiness(frame, model):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img, augment=False, size=640)  # Detect objects
    detections = non_max_suppression(results)
    
    alert = False
    for det in detections:
        if det is not None and len(det):
            for *xyxy, conf, cls in det:
                label = model.names[int(cls)]
                if label in ["closed_eyes", "yawn"]:
                    alert = True
                    cv2.putText(frame, "DROWSY!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # Draw bounding box
                xyxy = [int(x) for x in xyxy]
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame, alert

# Streamlit UI
st.title("Driver Drowsiness Detection System")

run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

model = load_model()

if run:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not detected.")
            break
        
        # Detect drowsiness
        frame, alert = detect_drowsiness(frame, model)
        
        # Display the video frame
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if alert:
            st.warning("Drowsiness Detected! Please take a break.")
else:
    st.write("Click the checkbox above to start the camera.")
