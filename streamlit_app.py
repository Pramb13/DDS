import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define video processor class
class VideoProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frm, (x, y), (x + w, y + h), (0, 255, 0), 3)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# Streamlit App UI
st.title("🚘 Live Driver Drowsiness Detection")
st.markdown("Real-time face detection using Streamlit WebRTC.")

# Start webcam streaming
webrtc_streamer(
    key="drowsiness-detection",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
)
