import cv2
from ultralytics import YOLO
import streamlit as st
import tempfile
import os

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Streamlit UI setup
st.title("Object Detection on Uploaded Video with YOLOv8")
st.write("Upload a video to start processing.")

# File uploader for video
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if video_file is not None:
    # Save the uploaded video temporarily
    temp_dir = tempfile.TemporaryDirectory()
    video_path = os.path.join(temp_dir.name, video_file.name)
    with open(video_path, "wb") as f:
        f.write(video_file.read())
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
   
    if st.sidebar.button("Start Processing"):
         
        if not cap.isOpened():
            st.error("Error opening video file.")
    
        else:
            st.sidebar.write("Processing video...")
            
            # Create a placeholder for video frames
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
                
                # Display the frame in Streamlit
                frame_placeholder.image(image, channels="RGB", use_container_width=True)
            
            cap.release()
        
        temp_dir.cleanup()