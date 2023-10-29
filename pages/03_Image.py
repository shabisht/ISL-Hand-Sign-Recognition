import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
from PIL import Image

st.title("ISL")

img = st.camera_input("Capture a photo with Indian Sign Language Digit or Alphabet", key='input_img')
if img is not None:
    bytes_data = img.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    # st.write(type(cv2_img))
    # st.write(cv2_img.shape)
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    # st.write(rgb_img.shape)
    # st.write(type(rgb_img))
    results = st.session_state.hands.process(rgb_img)
    if results.multi_hand_landmarks:
        height, width, _ = rgb_img.shape
        offset = 20
        rgb_img2 = rgb_img.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            # print("hand")
            mp.solutions.drawing_utils.draw_landmarks(rgb_img2, hand_landmarks, connections=mp.solutions.hands.HAND_CONNECTIONS )
            x_min, y_min, x_max, y_max = np.inf, np.inf, 0, 0
            for landmark in hand_landmarks.landmark:
                x_min, y_min = int(min(landmark.x*width, x_min)), int(min(landmark.y*height, y_min))
                x_max, y_max = int(max(landmark.x*width, x_max)), int(max(landmark.y*height, y_max))
    
            # cv2.imwrite("test.jpg", img[y_min-offset:y_max+offset, x_min-offset:x_max+offset])
            try:
                test_img = Image.fromarray(rgb_img[y_min-offset:y_max+offset, x_min-offset:x_max+offset])
                test_img = test_img.resize((250,250))
                test_img = np.expand_dims(np.array(test_img), axis=0)
                prediction = st.session_state.model.predict(test_img)
                value = max(prediction[0])
                label = np.argmax(prediction[0])

                cv2.rectangle(rgb_img2, (x_min-offset, y_min-offset), (x_max+offset, y_max+offset), (255,0,0),2)
                cv2.putText(rgb_img2, f"{st.session_state.class_labels[label]} - {round(value*100,2)}%", (x_min, y_min-offset-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            except ValueError:
                pass
        col1, col2 = st.columns(2)
        col1.header("Orignal Image")
        col1.image(rgb_img)
        col2.header("Hand Sign Detected")
        col2.image(rgb_img2)
    else:
         st.image(img)
         st.warning("Sorry no hands detected in above image")