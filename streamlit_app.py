import streamlit as st
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

# Constants for EAR threshold and consecutive frame count
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Load pre-trained face and landmark detectors
def load_detectors():
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(dlib.dat_file("shape_predictor_68_face_landmarks.dat"))
    return face_detector, landmark_predictor

# Main Streamlit app
def main():
    st.title("Driver Drowsiness Detection System")
    st.sidebar.header("Settings")
    EAR_THRESHOLD = st.sidebar.slider("EAR Threshold", 0.2, 0.3, 0.25, step=0.01)
    CONSEC_FRAMES = st.sidebar.slider("Consecutive Frames for Alert", 10, 30, 20)

    st.write("""
        This system uses your webcam to monitor eye movements and detect drowsiness. If your eyes remain closed for too long, the system triggers an alert.
        """)

    # Load face and landmark detectors
    face_detector, landmark_predictor = load_detectors()

    # Start video capture
    video_capture = cv2.VideoCapture(0)
    (lStart, lEnd) = (42, 48)
    (rStart, rEnd) = (36, 42)

    frame_counter = 0
    drowsy = False

    stframe = st.empty()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            st.error("Failed to access webcam. Make sure it's connected.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)

        for face in faces:
            shape = landmark_predictor(gray, face)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            left_eye = shape[lStart:lEnd]
            right_eye = shape[rStart:rEnd]

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)

            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= CONSEC_FRAMES:
                    drowsy = True
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                frame_counter = 0
                drowsy = False

            # Draw contours
            left_hull = cv2.convexHull(left_eye)
            right_hull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [left_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_hull], -1, (0, 255, 0), 1)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
