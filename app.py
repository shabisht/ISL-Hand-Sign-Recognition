import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode
# import mediapipe as mp
# import tensorflow.keras as keras
from PIL import Image
import av
import logging
import os
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client


st.set_page_config(page_title='ISL Detection', page_icon=':clapper:')

# @st.cache_resource
# def load_model():
#     model = keras.models.load_model('models/ISL2.keras')
#     return model

class_labels = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5',
                6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11:
                'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16:
                'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21:
                'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26:
                'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31:
                'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'}

hands = mp.solutions.hands.Hands(
                static_image_mode= False,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
                )

# class VideoTransformer(VideoTransformerBase):
#     def __init__(self):
#         self.i = 0
#     def transform(self, frame):
#         img = frame.to_ndarray(format="bgr24")
#         height, width, _ = img.shape
#         offset = 30
#         results = hands.process(img)
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp.solutions.draw_utis.draw_landmarks(
#                     img, hand_landmarks, connections=mp.solutions.hands.HAND_CONNECTIONS
#                 )
#                 x_min, y_min, x_max, y_max = np.inf, np.inf, 0, 0
#                 for landmark in hand_landmarks.landmark:
#                     x_min, y_min = int(min(landmark.x*width, x_min)), int(min(landmark.y*height, y_min))
#                     x_max, y_max = int(max(landmark.x*width, x_max)), int(max(landmark.y*height, y_max))
        
#             # cv2.imwrite("test.jpg", img[y_min-offset:y_max+offset, x_min-offset:x_max+offset])
#             try:
#                 test_img = Image.fromarray(img[y_min-offset:y_max+offset, x_min-offset:x_max+offset])
#                 test_img = test_img.resize((250,250))
#                 test_img = np.expand_dims(np.array(test_img), axis=0)
#                 prediction = model.predict(test_img)
#                 value = max(prediction[0])
#                 label = np.argmax(prediction[0])

#                 cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255,0,0),2)
#                 cv2.putText(img, f"{label}- {value.format('.2f')}", (x_min, y_min-offset-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            
#             except ValueError:
#                 pass
#         img = cv2.flip(img, 1)
#         return img
    
# model = load_model()

st.header("Indian Sign Language Symbol Detection")
with st.sidebar:
    st.info("Test")

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    height, width, _ = img.shape
    offset = 30
    # results = hands.process(img)
    # if results.multi_hand_landmarks:
    #     for hand_landmarks in results.multi_hand_landmarks:
    #         mp.solutions.draw_utis.draw_landmarks(
    #             img, hand_landmarks, connections=mp.solutions.hands.HAND_CONNECTIONS
    #         )
    #         x_min, y_min, x_max, y_max = np.inf, np.inf, 0, 0
    #         for landmark in hand_landmarks.landmark:
    #             x_min, y_min = int(min(landmark.x*width, x_min)), int(min(landmark.y*height, y_min))
    #             x_max, y_max = int(max(landmark.x*width, x_max)), int(max(landmark.y*height, y_max))
    
    #     # cv2.imwrite("test.jpg", img[y_min-offset:y_max+offset, x_min-offset:x_max+offset])
    #     try:
    #         test_img = Image.fromarray(img[y_min-offset:y_max+offset, x_min-offset:x_max+offset])
    #         test_img = test_img.resize((250,250))
    #         test_img = np.expand_dims(np.array(test_img), axis=0)
    #         prediction = model.predict(test_img)
    #         value = max(prediction[0])
    #         label = np.argmax(prediction[0])

    #         cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255,0,0),2)
    #         cv2.putText(img, f"{label}- {value.format('.2f')}", (x_min, y_min-offset-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        
    #     except ValueError:
    #         pass

    return av.VideoFrame.from_ndarray(img, format="bgr24")


# model = load_model()

webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)   

