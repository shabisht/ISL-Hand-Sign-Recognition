import streamlit as st

st.set_page_config(page_title='ISL Detection', layout='wide', page_icon=':clapper:')

# cols = st.columns(3)
# cols[1].image('images/ISL Logo2.jpg', use_column_width=True)
st.image('images/ISL Logo2.jpg')

message = '''
        # Indian Sign Language ☝️👎🖐️
        - Indian Sign Language (ISL) is recognized as one of the official languages
         of India alongside spoken languages like Hindi, English, and others.
        - ISL has its own distinct vocabulary and grammar system. It is not directly based on spoken languages like Hindi or English but has its own
         syntax and grammar rules. 
        - ISL relies on various handshapes and movements to convey meaning. Different handshapes and movements can represent different words, ideas, or
        concepts.
        # Hand Signs for Digits 0-9
            '''
st.markdown(message, unsafe_allow_html=True)
st.image("images/ISL-digits.jpg")

st.info("👈 visit ISL-Detector")