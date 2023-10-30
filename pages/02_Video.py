# import cv2
# import streamlit as st

# st.title("Webcam Live Feed")
# run = st.checkbox('Run')
# FRAME_WINDOW = st.image([])
# camera = cv2.VideoCapture(0)

# while run:
#     _, frame = camera.read()
#     cv2.rectangle(frame, (100,100), (400,400),(0,255,0),2)
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     FRAME_WINDOW.image(frame)
# else:
#     st.write('Stopped')


import cv2
import numpy as np
import av
import mediapipe as mp
from PIL import Image
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def process(image):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height, width, _ = img.shape
    offset = 20
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            img,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
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
    return img


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = process(img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)