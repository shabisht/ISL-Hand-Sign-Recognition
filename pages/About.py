import streamlit as st

st.set_page_config(page_title='ISL Detection', layout='wide', page_icon=':clapper:')

cols = st.columns([2.5,5,2.5])
cols[1].image('images/ISL Logo1.png')

message = '''
        # Objective
         The objective of this web app is to develop a real-time Indian Sign Language (ISL) symbol
          detection system. Its primary aim is to facilitate seamless communication between 
          hearing-impaired individuals and the general population. Many people are not familiar 
          with sign language, which can create communication barriers. To address this, we are 
          initially focusing on detecting basic hand signs related to alphabets and digits. Our 
          goal is to ensure that hearing-impaired individuals can be better understood by others, 
          enhancing their overall communication experience.

        # Indian Sign Language 🖐️ (ISL) History
        In the 2000s, the Indian deaf community advocated for an institute dedicated to Indian 
        Sign Language (ISL) teaching and research. Their efforts culminated in the approval of 
        the Indian Sign Language Research and Training Center (ISLRTC) under the Indira Gandhi 
        National Open University (IGNOU), Delhi, in 2011. However, the center at IGNOU closed 
        in 2013, sparking protests. Following discussions and protests, ISLRTC was integrated
          with the Ali Yavar Jung National Institute of Hearing Handicapped (AYJNIHH) in 2015. 
          After further negotiations, ISLRTC was officially established as a Society under the 
          Department of Empowerment of Persons with Disabilities, Ministry of Social Justice
        and Empowerment, in New Delhi on September 28, 2015. This achievement significantly
        addressed gaps in the education and communication needs of India's deaf community.

        # ISL Representation
        - ISL has its own distinct vocabulary and grammar system. It is not directly based on spoken languages like Hindi or English but has its own
         syntax and grammar rules. 
        - ISL relies on various handshapes and movements to convey meaning. Different handshapes and movements can represent different words, ideas, or
        concepts.
        # Hand Signs for Digits 0-9
            '''
st.markdown(message, unsafe_allow_html=True)
st.image("images/ISL-digits.jpg")

st.markdown("# Hand Signs for Alphabet A-Z", unsafe_allow_html=True)
st.image("images/ISL alphabets.jpg")