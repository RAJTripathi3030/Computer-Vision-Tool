import streamlit as st
import cv2 
from ultralytics import YOLO
import tempfile
import os

st.title("Object Tracking on Uploaded Video with YOLOv8")

video_file = st.file_uploader("Upload a video to start processing.", type=["mp4", "avi", "mov", "mkv"])
if video_file is not None:
    temp_dir = tempfile.TemporaryDirectory()
    video_path = os.path.join(temp_dir.name, video_file.name)
    with open(video_path, "wb") as f:
        f.write(video_file.read())
        
    model = YOLO("yolov8n.pt")
    
    
    cap = cv2.VideoCapture(video_path)
    tracker = cv2.TrackerKCF_create()
    
    if st.sidebar.button("Start Processing"):
        if not cap.isOpened():
            st.error("Error opening video file.")
        else:
            st.sidebar.write("Processing video...")
            frame_placeholder = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = model(frame)
                annotated_frame = results[0].plot()
                
                image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                frame_placeholder.image(image, channels="RGB", use_container_width=True)
            cap.release()
            
        temp_dir.cleanup()