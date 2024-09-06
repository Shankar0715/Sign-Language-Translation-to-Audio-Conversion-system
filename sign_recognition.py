import cv2
import threading
import pyttsx3
import streamlit as st
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Thread-safe pyttsx3 engine initialization function
def init_tts_engine():
    engine = pyttsx3.init()
    return engine


# Text-to-speech function
def text_to_speech(text):
    engine = init_tts_engine()
    engine.say(text)
    engine.runAndWait()


# Sign recognition function
def recognize_sign(model_dict, hands):
    model = model_dict["model"]
    labels_dict = model_dict["labels_dict"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open video stream.")
        return

    # Create columns for layout
    col1, col2 = st.columns([2, 1])  # 2:1 ratio for left and right columns
    frame_display = col1.empty()
    prediction_display = col2.empty()

    stop_button = st.empty()
    stop = stop_button.button("Stop Recognition")

    # Recognized text
    recognized_text = ""
    frame_counter = 0  # Counter to skip frames
    frame_skip = 2  # Number of frames to skip for processing

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        if frame_counter % frame_skip != 0:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            if len(data_aux) == model.n_features_in_:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                recognized_text = predicted_character

                prediction_display.markdown(
                    f"<h1 style='text-align: center; border: 2px solid black; padding: 10px;'>{recognized_text}</h1>",
                    unsafe_allow_html=True,
                )

                tts_thread = threading.Thread(
                    target=text_to_speech, args=(recognized_text,)
                )
                tts_thread.start()

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_display.image(frame, channels="RGB")

            if stop:
                break

    cap.release()
    cv2.destroyAllWindows()
    
