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
    uploaded_files = st.file_uploader(
        "Upload a template and a main image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )   
    if len(uploaded_files) == 2:
        # Open color images for display
        pil_template = Image.open(uploaded_files[0])
        pil_main = Image.open(uploaded_files[1])

        # Convert to NumPy arrays and grayscale for processing
        template_image = cv2.cvtColor(np.array(pil_template), cv2.COLOR_RGB2GRAY)
        main_image = cv2.cvtColor(np.array(pil_main), cv2.COLOR_RGB2GRAY)
        
        col1, col2 = st.columns(2)
        with col1:
            # Resize template if it's larger than the main image
            if template_image.shape[0] > main_image.shape[0] or template_image.shape[1] > main_image.shape[1]:
                scale_factor = min(main_image.shape[0] / template_image.shape[0], 
                                   main_image.shape[1] / template_image.shape[1])
                new_size = (int(template_image.shape[1] * scale_factor), 
                            int(template_image.shape[0] * scale_factor))
                template_image = cv2.resize(template_image, new_size)
           
            w, h = template_image.shape[::-1] # Get the width and height of the template image
            
            # Display the ORIGINAL color image, not the grayscale
            st.image(pil_template, caption="Template Image", use_container_width=True)
        
        with col2:
            # Display the ORIGINAL color image, not the grayscale
            st.image(pil_main, caption="Main Image", use_container_width=True)
        
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
        matching_option = st.sidebar.selectbox("", list(matching_methods.keys()))
        
        if st.sidebar.button("Match Images") and matching_option: 
            method = matching_methods[matching_option]
            result = cv2.matchTemplate(main_image, template_image, getattr(cv2, method))
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if method in ["TM_SQDIFF", "TM_SQDIFF_NORMED"]:
                top_left = min_loc
            else:
                top_left = max_loc
            
            bottom_right = (top_left[0] + w, top_left[1] + h)
            
            # Convert main image to color for drawing rectangle
            final_image = cv2.cvtColor(main_image, cv2.COLOR_GRAY2RGB)
            cv2.rectangle(final_image, top_left, bottom_right, 255, 10)

            # Convert final image back to a format Streamlit can display
            final_pil_image = Image.fromarray(final_image)
            st.image(final_pil_image, caption="Matched Image", use_container_width=True)
        
    else:
        st.warning("Please upload exactly two images.")
