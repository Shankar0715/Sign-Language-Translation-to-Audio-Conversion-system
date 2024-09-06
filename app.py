import os
import pickle
import cv2
import numpy as np
import mediapipe as mp
import threading
import pyttsx3
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sign_recognition import recognize_sign  # Import the function from sign_recognition.py

# Constants
DATA_DIR = "./data"
DATASET_SIZE = 100

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Streamlit UI
def main():
    st.title("Sign Language Recognition")

    # Section for Data Collection
    st.header("1. Collect Data")
    class_name = st.text_input("Enter the class name:")

    if st.button("Start Capture"):
        if class_name:
            capture_images(class_name)
        else:
            st.error("Please enter a class name.")

    # Section for Dataset Creation
    st.header("2. Create Dataset")
    if st.button("Create Dataset"):
        create_dataset()

    # Section for Model Training
    st.header("3. Train Model")
    if st.button("Train Model"):
        train_model()

    # Section for Sign Recognition
    st.header("4. Recognize Sign")
    if st.button("Start Recognition"):
        model_dict = pickle.load(open("model.p", "rb"))
        recognize_sign(model_dict, hands)


def capture_images(class_name):
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    cap = cv2.VideoCapture(0)
    st.text(f"Collecting data for class '{class_name}'")

    done = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.putText(
            frame,
            f'Collecting data for "{class_name}". Ready? Press "Q"!',
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("frame", frame)
        if cv2.waitKey(25) == ord("q"):
            break
    counter = 0
    while counter < DATASET_SIZE:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame.")
            break

        cv2.putText(
            frame,
            f'Collecting data for "{class_name}": {counter}/{DATASET_SIZE}',
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("frame", frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, "{}.jpg".format(counter)), frame)
        counter += 1
        st.text(f"Image {counter}/{DATASET_SIZE} captured.")

    cap.release()
    cv2.destroyAllWindows()
    # Your existing code for capturing images here

def create_dataset():
    data = []
    labels = []

    for idx, dir_ in enumerate(os.listdir(DATA_DIR)):
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            data_aux = []
            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
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

                data.append(data_aux)
                labels.append(idx)  # Append numeric label

    with open("data.pickle", "wb") as f:
        pickle.dump({"data": data, "labels": labels}, f)
    st.success("Dataset created and saved!")
    # Your existing code for creating dataset here


def train_model():
    data_dict = pickle.load(open("data.pickle", "rb"))

    data = np.asarray(data_dict["data"])
    labels = np.asarray(data_dict["labels"])

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels
    )

    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)

    with open("model.p", "wb") as f:
        pickle.dump(
            {
                "model": model,
                "labels_dict": {i: dir_ for i, dir_ in enumerate(os.listdir(DATA_DIR))},
            },
            f,
        )

    st.success(f"{score * 100:.2f}% of samples were classified correctly!")

    # Your existing code for training model here


if __name__ == "__main__":
    main()
