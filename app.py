import cv2
import streamlit as st
import numpy as np
from keras._tf_keras.keras.preprocessing.image import img_to_array
from keras._tf_keras.keras.models import load_model
from data import Output

def toggle_start():
    st.session_state.start = True
    st.session_state.stop = False

def toggle_stop():
    st.session_state.stop = True
    st.session_state.start = False

def save_frame():
    st.session_state.save = True


# Load Model
model = load_model("./model/best_model.keras")
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize session state variables
if 'start' not in st.session_state:
    st.session_state.start = False
if 'stop' not in st.session_state:
    st.session_state.stop = False
if 'save' not in st.session_state:
    st.session_state.save = False

st.title("FACIAL EMOTION DETECTION")

frame_placeholder = st.empty()
col1, col2, col3, col4, col5 = st.columns(5)

with col2:
    st.button("Start", on_click=toggle_start)
with col3:
    st.button("Stop", on_click=toggle_stop)
with col4:
    st.button("Save", on_click=save_frame)


if st.session_state.start and not st.session_state.stop:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("The video capture has ended")
            break
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
            roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
            roi_gray = cv2.resize(roi_gray, (224, 224))
            img_pixels = img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            predictions = model.predict(img_pixels)

            # find max indexed array
            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]
            cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        
        frame_placeholder.image(frame, channels="RGB")

        if st.session_state.save:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            cv2.imwrite("frame.png", frame)
            st.session_state.save = False  # Reset save flag after saving

        if st.session_state.stop:
            frame_placeholder.empty()
            break

    cap.release()
    cv2.destroyAllWindows()



