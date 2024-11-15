
import streamlit as st
import cv2
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist
import numpy as np
import cvlib as cv
from cvlib.object_detection import draw_bbox
import time

st.title("Student Focus Tracker")
st.text("Analyzes eye and head movements to determine focus and distractions, including phone usage.")

# Initialize dlib's face detector and the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to calculate the eye aspect ratio
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to track head position
def is_head_straight(nose):
    (x, y) = nose[3]
    return 250 < x < 350

# Streamlit elements
frame_placeholder = st.empty()
info_placeholder = st.empty()

# Initialize variables
phone_pick_count = 0
distraction_count = 0
focus_start_time = None
focus_durations = []
head_positions = []

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48
HEAD_POSITION_CONSEC_FRAMES = 48

COUNTER = 0
TOTAL = 0
HEAD_COUNTER = 0

# Camera control buttons
start_button = st.button("Start Camera")
stop_button = st.button("Stop Camera")

if start_button:
    cap = cv2.VideoCapture(0)
    focus_start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][1]]
            rightEye = shape[face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][1]]
            nose = shape[face_utils.FACIAL_LANDMARKS_IDXS["nose"][0]:face_utils.FACIAL_LANDMARKS_IDXS["nose"][1]]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            if ear < EYE_AR_THRESH:
                COUNTER += 1
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                    distraction_count += 1
                COUNTER = 0

            if is_head_straight(nose):
                HEAD_COUNTER += 1
            else:
                if HEAD_COUNTER >= HEAD_POSITION_CONSEC_FRAMES:
                    distraction_count += 1
                HEAD_COUNTER = 0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(nose)], -1, (255, 0, 0), 1)

        # Phone detection
        bbox, label, conf = cv.detect_common_objects(frame, confidence=0.25, model='yolov3-tiny')
        frame = draw_bbox(frame, bbox, label, conf)

        if 'cell phone' in label:
            phone_pick_count += 1
            distraction_count += 1

        frame_placeholder.image(frame, channels="BGR")
        info_placeholder.text(f"Blinks: {TOTAL} | Phone Picks: {phone_pick_count} | Distractions: {distraction_count}")

        if stop_button:
            cap.release()
            break

    # Calculate focus durations
    focus_end_time = time.time()
    focus_duration = focus_end_time - focus_start_time if focus_start_time else 0
    focus_durations.append(focus_duration)

    # Report
    longest_focus = max(focus_durations) if focus_durations else 0
    report = f'''

    **Focus Analysis Report:**
    - Number of times phone picked: {phone_pick_count}
    - Number of distractions: {distraction_count}
    - Longest focus time: {longest_focus:.2f} seconds
    '''
    st.markdown(report)
    st.success("Focus analysis completed!")

