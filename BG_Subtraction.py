import streamlit as st
import cv2 as cv 
import tempfile
import requests
from PIL import Image
import numpy as np

# Set the page configuration
# st.set_page_config(page_title="Background Subtraction Explained", layout="wide")

tab1, tab2 = st.tabs(['About Background Subtraction', 'Implementation'])
with tab1:
    
    # Title and introduction
    st.title("Background Subtraction Explained")
    st.write("""
    Background subtraction is a computer vision technique used to separate moving objects from the static background in a video.
    Imagine you have a security camera watching an empty parking lot. The system first learns what the empty lot looks like 
    and then detects changes (like a car entering) by comparing each new frame to the learned background.
    """)

    # Section: How it Works
    st.header("How Does It Work?")
    st.markdown("""
    1. **Background Modeling:**  
    - **Initialization:** The system captures an initial image (or series of images) to model the static background.  
    - **Update:** Over time, the background model is updated to reflect gradual changes (e.g., lighting, weather).
    
    2. **Foreground Detection:**  
    - Each new video frame is compared against the background model.  
    - The differences (i.e., moving objects) are highlighted in a binary image called the *foreground mask*.
    
    3. **Result:**  
    - The foreground mask is a binary image where white pixels represent moving objects (e.g., people, cars) and black pixels represent the background.
    """)

    # Display an example image (replace with your own link if necessary)
    image = Image.open(r"https://learnopencv.com/moving-object-detection-with-opencv/")
    st.image(image, caption="Example: Moving objects (white) vs. background (black)", use_container_width=True)

    # Section: Real-world Example
    st.header("Real-world Example")
    st.write("""
    Consider a video from a security camera in a parking lot:
    - The **background model** is the image of the empty parking lot.
    - When a car enters, the difference between the current frame and the background model is computed.
    - The result is a mask highlighting the car as the moving object.
    """)

    # Sidebar for extra information and navigation
    st.sidebar.title("Learn More")
    st.sidebar.info("Use the interactive slider below to conceptually adjust a threshold value for detecting differences between the frame and the background model.")

    # Interactive element: slider (for conceptual demonstration)
    threshold = st.sidebar.slider("Threshold value (conceptual)", min_value=0, max_value=100, value=50)
    st.sidebar.write(f"You selected a threshold of: **{threshold}**")

    # Explanation conclusion
    st.markdown("### Summary")
    st.write("""
    Background subtraction is essential in video processing applications such as surveillance and traffic monitoring. 
    It works by first learning the static background and then detecting moving objects by comparing new frames against this model.
    """)
    
with tab2:
    st.header("Real-Time Background Subtraction")
    st.write("""
    The left column displays the original video, while the right column shows the processed foreground mask (obtained by background subtraction) in real time.
    """)

    # URL for the sample video (AVI format)
    video_url = "https://github.com/opencv/opencv/raw/master/samples/data/vtest.avi"

    @st.cache_resource
    def download_video(url: str) -> str:
        response = requests.get(url)
        # Save the video to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
        temp_file.write(response.content)
        temp_file.close()
        return temp_file.name

    video_path = download_video(video_url)

    # Open the video file using OpenCV
    capture = cv.VideoCapture(video_path)
    if not capture.isOpened():
        st.error("Unable to open video file!")
    else:
        st.success("Video file opened successfully.")
        backSub = cv.createBackgroundSubtractorMOG2()

        # Create two columns for simultaneous display of original and processed frames
        col1, col2 = st.columns(2)
        original_placeholder = col1.empty()
        processed_placeholder = col2.empty()

        # Process and display video frames in real time
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break
            # Apply background subtraction to get the foreground mask
            mask = backSub.apply(frame)

            # Convert BGR to RGB for proper color display in Streamlit
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            # Convert the grayscale mask to RGB for display consistency
            mask_rgb = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)

            # Update the placeholders with the new frames
            original_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            processed_placeholder.image(mask_rgb, channels="RGB", use_container_width=True)

        capture.release()
    