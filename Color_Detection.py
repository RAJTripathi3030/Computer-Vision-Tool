import streamlit as st
import cv2
import numpy as np
from PIL import Image
import imutils
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

tab1, tab2 = st.tabs(['About Color Detection', 'Implementation'])

with tab1:
        st.title("Color Detection Using OpenCV")
        st.write("""
                This page will demonstrate how to detect colors in an image using OpenCV.
                """)

        st.header("What is Color Detection?")
        st.write("""
                Color detection is a computer vision technique used to identify and extract specific colors from an image or video.
                OpenCV provides the tools to convert images to different color spaces (like RGB or HSV) and apply thresholding 
                methods to isolate the desired colors.
                """)

        st.header("Why is it Useful?")
        st.write("""
                - **Object Tracking and Recognition:** Enables the detection and tracking of objects based on their color.
                - **Image Segmentation:** Helps in isolating objects from the background by differentiating colors.
                - **Automation:** Supports various automated systems in industries such as manufacturing and traffic management.
                """)

        st.header("Where is it Used?")
        st.write("""
                - **Traffic Systems:** Detecting traffic signals and road signs.
                - **Medical Imaging:** Enhancing analysis by focusing on specific color patterns in images.
                - **Industrial Quality Control:** Identifying product features and defects.
                - **Augmented Reality:** Overlaying digital information onto real-world objects based on their colors.
                """)
        
with tab2:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
                pil_image = Image.open(uploaded_file).convert("RGB")
                image_HSV = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2HSV)
                image = np.array(pil_image)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                if st.button("View Colors"):
                        hist = cv2.calcHist([image_HSV], [0], None, [180], [0, 180])
                        peaks = np.argsort(hist.ravel())[-4:]
                        dominant_colors = [cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2RGB)[0][0] for hue in peaks]
                        fig, ax = plt.subplots(figsize=(10, 2))
                        for i, color in enumerate(dominant_colors):
                                ax.fill_between([i, i+1], 0, 1, color=color / 255)
                                ax.axis("off")
                                ax.set_title("Dominant Colors")
                        st.pyplot(fig)
                