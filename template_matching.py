import streamlit as st 
import numpy as np
import cv2 
from PIL import Image

tab1, tab2 = st.tabs(["About Template Matching", "Implementation"])

with tab1:
    st.header("What is Template Matching?")
    st.write(
        "Template matching is a technique in computer vision used to find a small part of an image that matches a template image. "
        "It works by sliding the template over the input image and comparing the similarity at each position. "
        "Common similarity metrics include cross-correlation and squared differences."
    )
    
    st.subheader("How It Works")
    st.write(
        "1. A smaller template image is selected.\n"
        "2. The template is compared to different regions of the larger image.\n"
        "3. A similarity score is calculated for each region.\n"
        "4. The region with the highest similarity is identified as the match."
    )
    
    st.subheader("Applications")
    st.write(
        "- Object detection\n"
        "- Image alignment\n"
        "- Quality inspection in manufacturing\n"
        "- Medical image analysis"
    )

with tab2:
    st.write("This tab is currently empty.")
    uploaded_files = st.file_uploader(
        "Upload a template and a main image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )   
    if uploaded_files and len(uploaded_files) == 2:
        template_image, main_image = uploaded_files
        col1, col2 = st.columns(2)
        with col1:
            template_image = cv2.cvtColor(np.array(Image.open(template_image)), cv2.COLOR_RGB2GRAY)
            st.image(template_image, caption="Template Image", use_container_width=True)
        with col2:
            main_image = cv2.cvtColor(np.array(Image.open(main_image)), cv2.COLOR_RGB2GRAY)
            st.image(main_image, caption="Main Image", use_container_width=True)
    else:
        st.warning("Please upload exactly two images.")
        
        