import streamlit as st
from pytube import YouTube
from ultralytics import YOLO
import os
import cv2
from tqdm import trange
import torch

def process(video_path):
    torch.cuda.set_device(0)
    st.sidebar.info("Torch cuda is available: " + str(torch.cuda.is_available()))
    video_path = video_path.replace('\\', '/')  

    if not os.path.exists('./results'):
        os.makedirs('./results')
    output_video_path = f'./results/process_{os.path.basename(video_path)}'

    cap = cv2.VideoCapture(video_path)
    output_fps = int(cap.get(cv2.CAP_PROP_FPS)) 
    output_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    output_video = cv2.VideoWriter(output_video_path, fourcc, output_fps, output_size)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model = YOLO("yolov8n.pt")
    progress_bar = st.progress(0)

    for index in trange(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        progress_bar.progress(index / frame_count)

        results = model.predict(frame, verbose=False)
        name = results[0].names

        cls = results[0].boxes.cls
        id = torch.unique(cls, return_counts=True)[0].cpu().detach().numpy()
        count = torch.unique(cls, return_counts=True)[1].cpu().detach().numpy()

        annotated_frame = results[0].plot()
        dic = dict(zip(id, count))
        for idx, i in enumerate(dic.keys(), start = 1):
            cv2.putText(annotated_frame, str(str(name[i]) + ": " + str(dic[i])), (50, 50 *idx), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (36, 255, 12), 3)

        output_video.write(annotated_frame)

    progress_bar.progress(100)
    
    output_video.release()
    cap.release()
    st.video(output_video_path)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Video Analytics System",
        page_icon="📹",
        layout="wide",
        initial_sidebar_state="auto"
    )
    st.title("Video Analytics System (VAS)")

    choice = st.sidebar.radio("Upload video from", ["URL link", "Your computer"], index=0)

    save_file = os.getcwd()
    if not os.path.exists('./download_video'):
        os.makedirs('./download_video')
    save_file = os.path.join(save_file, 'download_video')

    if choice == "Your computer":
        st.subheader("Upload a video from your computer:")
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi"])

        if uploaded_file:
            file_name = uploaded_file.name
            file_path = os.path.join(save_file, file_name)
            
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.read())
            
            if st.button("Analyzing", key="analyze_button"):
                process(file_path)
    else:
        st.subheader("Analyze a video from a URL:")
        url = st.text_input('Enter the URL link')
        
        if st.button("Analyzing", key="analyze_button"):
            yt = YouTube(url)
            stream = yt.streams.get_highest_resolution()
            download_path = os.path.join(save_file, stream.default_filename)
            stream.download(output_path=save_file)
            process(download_path)
