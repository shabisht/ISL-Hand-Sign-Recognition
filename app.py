import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import tensorflow.keras as keras

st.set_page_config(page_title='ISL Detection', page_icon=':clapper:')

@st.cache_resource
def load_model():
    model = keras.models.load_model('models/ISL4.keras')
    return model

if 'hands' not in st.session_state:
    st.session_state.hands = mp.solutions.hands.Hands(
                                static_image_mode= False,
                                max_num_hands=2,
                                model_complexity=1,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5
                                )
    st.session_state.model = load_model()
    st.session_state.class_labels = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5',
                                     6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11:
                                    'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16:
                                    'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21:
                                    'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26:
                                    'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31:
                                    'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'}



