def process_video(st, np, cv2, mp, Image):
    video_file = st.file_uploader('video', type =['mp4','mov', 'avi'])
    # cap = cv2.VideoCapture(video_file)
    if video_file is not None:
        play = st.toggle("Play video")
        if play:
            st.video(video_file)
    # while True:
    #     success, frame = cap.read()
    #     if success:


