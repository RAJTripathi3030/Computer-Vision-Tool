import cv2 
import streamlit as st 
from ultralytics import YOLO 
import tempfile 
import os

# Load the YOLOc8 Segmentation Model 
model = YOLO("yolov8n-seg.pt")

st.title("Object Segmentation on Uploaded Video with YOLOv8")

# Uploaded File or Video 
video_file = st.file_uploader("Upload a video to start processing.", type=["mp4", "avi", "mov", "mkv"])
if video_file is not None:
    # Create a temporary file
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(video_file.read())
    temp_video.close()  # Close so OpenCV can access it
        
    # Init a vid capture model 
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
                
                # Perform inference
                results = model(frame)
                annotated_frame = results[0].plot() 
                
                # Convert to RGB for Streamlit
                image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                frame_placeholder.image(image, channels="RGB", use_container_width=True)
            cap.release()
            
         # Delete the temporary file after processing
        os.remove(temp_video.name)