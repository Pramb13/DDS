import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(landmarks):
    # Define the indices for the left and right eye landmarks
    left_eye_indices = [362, 385, 387, 263, 373, 380]
    right_eye_indices = [33, 160, 158, 133, 153, 144]
    
    # Calculate EAR for both eyes
    left_ear = calculate_single_ear(landmarks, left_eye_indices)
    right_ear = calculate_single_ear(landmarks, right_eye_indices)
    
    return (left_ear + right_ear) / 2.0

def calculate_single_ear(landmarks, indices):
    # Calculate distances for EAR
    p1 = np.array([landmarks[indices[0]].x, landmarks[indices[0]].y])
    p2 = np.array([landmarks[indices[1]].x, landmarks[indices[1]].y])
    p3 = np.array([landmarks[indices[2]].x, landmarks[indices[2]].y])
    p4 = np.array([landmarks[indices[3]].x, landmarks[indices[3]].y])
    p5 = np.array([landmarks[indices[4]].x, landmarks[indices[4]].y])
    p6 = np.array([landmarks[indices[5]].x, landmarks[indices[5]].y])
    
    ear = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4))
    return ear

# Streamlit UI
st.title("Driver Drowsiness Detection")
st.write("This application detects drowsiness based on eye aspect ratio.")

# Video Stream
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    results = face_mesh.process(img)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ear = calculate_ear(face_landmarks.landmark)
            if ear < 0.2:  # Threshold for drowsiness
                cv2.putText(img, "DROWSY!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return img

webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
