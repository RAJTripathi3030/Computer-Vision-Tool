import streamlit as st
import cv2
from ultralytics import YOLO 
import tempfile
import os

# Load the YOLOv8 model
model = YOLO("yolov8n-pose.pt") 

st.title("Posture Detection on Uploaded Video with YOLOv8")

video_file = st.file_uploader("Upload a video to start processing.", type=["mp4", "avi", "mov", "mkv"]) 

if video_file is not None: 
   # Create a temporary file
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(video_file.read())
    temp_video.close()  # Close so OpenCV can access it
        
    # Open video capture
    cap = cv2.VideoCapture(temp_video.name)
    
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
        os.remove(temp_video.name)  # Remove the temporary file
        st.sidebar.write("Processing completed.")