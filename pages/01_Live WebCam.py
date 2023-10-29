import av
import cv2
import streamlit as st
import numpy as np
from PIL import Image
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import queue
from typing import NamedTuple, List

class Detection(NamedTuple):
    box: np.ndarray

result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    # print(img.shape)
    height, width, _ = img.shape
    offset = 30
    # print(img)
    results = st.session_state.hands.process(img)
    if results.multi_hand_landmarks:
        print("hand detected")
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(img, hand_landmarks, connections=mp.solutions.hands.HAND_CONNECTIONS )
            x_min, y_min, x_max, y_max = np.inf, np.inf, 0, 0
            for landmark in hand_landmarks.landmark:
                x_min, y_min = int(min(landmark.x*width, x_min)), int(min(landmark.y*height, y_min))
                x_max, y_max = int(max(landmark.x*width, x_max)), int(max(landmark.y*height, y_max))
    
        # cv2.imwrite("test.jpg", img[y_min-offset:y_max+offset, x_min-offset:x_max+offset])
        try:
            test_img = Image.fromarray(img[y_min-offset:y_max+offset, x_min-offset:x_max+offset])
            test_img = test_img.resize((250,250))
            test_img = np.expand_dims(np.array(test_img), axis=0)
            prediction = st.session_state.model.predict(test_img)
            value = max(prediction[0])
            label = np.argmax(prediction[0])

            cv2.rectangle(img, (0, 100), (0, 100), (255,0,0),2)

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255,0,0),2)
            cv2.putText(img, f"{st.session_state.class_labels[label]}- {value.format('.2f')}", (x_min, y_min-offset-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        
        except ValueError:
            pass

    detections = [
        Detection(
            box=(cv2.rectangle(img, (100,100), (400, 400), (255,0,0),2)),
        )
    ]

    result_queue.put(detections)
    return av.VideoFrame.from_ndarray(img, format="bgr24")


st.header("ISL")
webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)   