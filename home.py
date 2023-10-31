import cv2
import numpy as np
import av
import mediapipe as mp
from PIL import Image, ImageOps
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import tensorflow.keras as keras
from methods import *

@st.cache_resource
def load_model():
    model = keras.models.load_model("models/keras_model.h5", compile=False)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    class_labels = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5',
                6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11:
                'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16:
                'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21:
                'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26:
                'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31:
                'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'}
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
    return model, data, class_labels, mp_drawing, mp_drawing_styles, mp_hands, hands

st.set_page_config(page_title='ISL Detection', layout='wide', page_icon=':raised_hands:')
model, data, class_labels, mp_drawing, mp_drawing_styles, mp_hands, hands = load_model()

def process(image):
    image.flags.writeable = False
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = True
    # img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = hands.process(img_rgb)
    img = img_rgb.copy()
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
        x_min, y_min, x_max, y_max = np.inf, np.inf, 0, 0
        for landmark in hand_landmarks.landmark:
                x_min, y_min = int(min(landmark.x*width, x_min)), int(min(landmark.y*height, y_min))
                x_max, y_max = int(max(landmark.x*width, x_max)), int(max(landmark.y*height, y_max))
    
        # cv2.imwrite("test.jpg", img[y_min-offset:y_max+offset, x_min-offset:x_max+offset])
        try:
            test_img = Image.fromarray(img_rgb[y_min-offset:y_max+offset, x_min-offset:x_max+offset])
            test_img = ImageOps.fit(test_img, (224,224), Image.LANCZOS)
            test_img = np.asarray(test_img)
            # Normalize the image
            normalized_image_array = (test_img.astype(np.float32) / 127.5) - 1
            # Load the image into the array
            data[0] = normalized_image_array
            # test_img = np.expand_dims(np.array(test_img), axis=0)
            prediction = model.predict(data)
            value = max(prediction[0])
            label = np.argmax(prediction[0])
            cv2.rectangle(img, (x_min-offset, y_min-offset), (x_max+offset, y_max+offset), (255,0,0),2)
            cv2.putText(img, f"{class_labels[label]}- {round(value*100,2)}", (x_min, y_min-offset-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        
        except ValueError:
            pass
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = process(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def process_img(img):
        if img is not None:
            bytes_data = img.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_img)
            if results.multi_hand_landmarks:
                height, width, _ = rgb_img.shape
                offset = 20
                rgb_img2 = rgb_img.copy()
                # label, value = None, None
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
                    except:
                        offset=5
                        try:
                            test_img = Image.fromarray(rgb_img[y_min-offset:y_max+offset, x_min-offset:x_max+offset])
                        except:
                            offset=0
                            test_img = Image.fromarray(rgb_img[y_min-offset:y_max+offset, x_min-offset:x_max+offset])

                    test_img = ImageOps.fit(test_img, (224,224), Image.LANCZOS)
                    test_img = np.asarray(test_img)
                    # Normalize the image
                    normalized_image_array = (test_img.astype(np.float32) / 127.5) - 1
                    # Load the image into the array
                    data[0] = normalized_image_array
                    # test_img = np.expand_dims(np.array(test_img), axis=0)
                    prediction = model.predict(data)
                    value = max(prediction[0])
                    label = np.argmax(prediction[0])

                    cv2.rectangle(rgb_img2, (x_min-offset, y_min-offset), (x_max+offset, y_max+offset), (255,0,0),2)
                    cv2.putText(rgb_img2, f"{class_labels[label]} - {round(value*100,2)}%", (x_min, y_min-offset-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                    # st.header("Hand Sign Detected")
                    # st.image(rgb_img2)
                    # st.header("Hand Sign Detected")
                    st.image(rgb_img2)
                    st.subheader(f"Detected Digit is {class_labels[label]} - {round(value*100,2)}%")
            else:
                # st.image(img)
                st.warning("Sorry no hands detected in above image")


cols = st.columns([2.5,5,2.5])
cols[1].image('images/ISL Logo1.png')
st.header("Explore below options for ISL Digits Detection")

selected_option = st.selectbox("Select option", options=['WebCam', 'Upload Video','Upload Image'], key="selectOption")
if selected_option == 'WebCam':
    webCam_option = st.radio("select a mode", options=['Turn Off WebCam', 'Capture Image', 'Live WebCam'], key='webCam_radioOption')
    if webCam_option == 'Turn Off WebCam':
        pass
    elif webCam_option == 'Capture Image':
        img = st.camera_input("Capture a photo with Indian Sign Language Digit or Alphabet", key='input_img')
        process_img(img)
    else:
        webrtc_ctx = webrtc_streamer(
                        key="Live WebCam Hand Sign Detection",
                        mode=WebRtcMode.SENDRECV,
                        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                        media_stream_constraints={"video": True, "audio": False},
                        video_processor_factory=VideoProcessor,
                        async_processing=True)
        
elif selected_option == 'Upload Video':
    # uploadedVideo = st.file_uploader("Upload Video", help="Upload files are limited to 100MB", key="uploadedVideo")
    process_video(st, np, cv2, mp, Image)

else:
    img = st.file_uploader("Upload Image", help="Upload files are limited to 100MB", key="uploadedImage")
    process_img(img)
