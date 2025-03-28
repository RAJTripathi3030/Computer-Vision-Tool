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
            w, h = template_image.shape[::-1] # Get the width and height of the template image
            st.image(template_image, caption="Template Image", use_container_width=True)
        with col2:
            main_image = cv2.cvtColor(np.array(Image.open(main_image)), cv2.COLOR_RGB2GRAY)
            st.image(main_image, caption="Main Image", use_container_width=True)
        st.success("Please select the matching method from the sidebar to continue.")
        
        matching_methods = {
            "Correlation Coefficient": "TM_CCOEFF",
            "Normalized Correlation Coefficient": "TM_CCOEFF_NORMED",
            "Cross-Correlation": "TM_CCORR",
            "Normalized Cross-Correlation": "TM_CCORR_NORMED",
            "Sum of Squared Differences": "TM_SQDIFF",
            "Normalized Sum of Squared Differences": "TM_SQDIFF_NORMED"
        }

        
        st.sidebar.write("Select the matching method")
        matching_option = st.sidebar.selectbox("", ["Correlation Coefficient", "Normalized Correlation Coefficient", "Cross-Correlation", "Normalized Cross-Correlation", "Sum of Squared Differences", "Normalized Sum of Squared Differences"])
        
        if st.sidebar.button("Match Images") and matching_option: 
            method = matching_methods[matching_option]
            result = cv2.matchTemplate(main_image, template_image, getattr(cv2, method))
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if method in ["TM_SQDIFF", "TM_SQDIFF_NORMED"]:
                top_left = min_loc
            else:
                top_left = max_loc
            
            bottom_right = (top_left[0] + w, top_left[1] + h)
            final_image = cv2.rectangle(main_image, top_left, bottom_right, 255, 2)
            
            st.image(final_image, caption="Matched Image", use_container_width=True)
        
    else:
        st.warning("Please upload exactly two images.")
        