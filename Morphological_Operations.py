import streamlit as st
import cv2 
from PIL import Image
import numpy as np

tab1, tab2 = st.tabs(['About Morphological Operations', 'Implementation'])

with tab1:
    st.title("Morphological Operations in Computer Vision")

    st.write("""
    Morphological operations are image processing techniques that work on the structure of objects within an image. 
    They use a small shape called a structuring element to probe and modify the shapes in the image. 
    These methods are very useful for tasks like noise removal, shape analysis, and object extraction.
    """)

    st.header("Key Morphological Operations")

    st.subheader("Erosion")
    st.write("""
    Erosion shrinks bright regions (foreground objects). 
    It works by sliding a kernel over the image and removing pixels on object boundaries. 
    For example, it helps remove small white noise.
    """)

    st.subheader("Dilation")
    st.write("""
    Dilation is the opposite of erosion. 
    It adds pixels to the boundaries of objects, making them larger. 
    This operation can fill in small holes or gaps in an object.
    """)

    st.subheader("Opening")
    st.write("""
    Opening is a combination of erosion followed by dilation. 
    It is used to remove small objects or noise from the foreground while keeping the overall shape intact.
    """)

    st.subheader("Closing")
    st.write("""
    Closing is the reverse of opening (dilation followed by erosion). 
    It fills small holes or gaps within objects without significantly changing their overall shape.
    """)

    st.subheader("Morphological Gradient")
    st.write("""
    The morphological gradient is the difference between the dilated and eroded versions of an image. 
    It effectively highlights the edges of objects.
    """)

    st.subheader("Top Hat")
    st.write("""
    The Top Hat operation is defined as the difference between the original image and its opening. 
    It is useful for extracting small bright elements from an otherwise dark background.
    """)

    st.subheader("Black Hat")
    st.write("""
    The Black Hat operation is the difference between the closing of the image and the original image. 
    It highlights small dark regions on a bright background.
    """)

    st.write("""
    These operations are essential tools in computer vision to preprocess images, extract features, 
    and improve the performance of tasks such as object detection and segmentation.
    """)

with tab2: 
    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_img is not None: 
        col1, col2 = st.columns(2)
        with col1:
            pil_image = Image.open(uploaded_img).convert("RGB")
            image_HSV = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2HSV)
            image = np.array(pil_image)
            st.image(image, caption="Uploaded Image", width=300)
        
        with st.sidebar: 
            st.write("Select Morphological Operation")
            operation = st.selectbox("", ["Erosion", "Dilation", "Opening", "Closing", "Gradient", "Top Hat", "Black Hat"])
            kernel_size = st.slider("Kernel Size", 1, 10, 3)
            iterations = st.slider("Iterations", 1, 10, 1)
            
            if st.button("Apply Operation"):
                if operation == "Erosion":
                    image = cv2.erode(image, np.ones((kernel_size, kernel_size), np.uint8), iterations=iterations)
                elif operation == "Dilation":
                    image = cv2.dilate(image, np.ones((kernel_size, kernel_size), np.uint8), iterations=iterations)
                elif operation == "Opening":
                    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size), np.uint8), iterations=iterations)
                elif operation == "Closing":
                    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8), iterations=iterations)
                elif operation == "Gradient":
                    image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, np.ones((kernel_size, kernel_size), np.uint8), iterations=iterations)
                elif operation == "Top Hat":
                    image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, np.ones((kernel_size, kernel_size), np.uint8), iterations=iterations)
                elif operation == "Black Hat":
                    image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, np.ones((kernel_size, kernel_size), np.uint8), iterations=iterations)
                
                
                with col2: 
                    st.image(image, caption=f"{operation} Image", width=300)