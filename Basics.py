import streamlit as st
import cv2
import numpy as np
from PIL import Image
import imutils

# Title and description
st.title("OpenCV Basics of Dealing with Images")
st.write("""
This Streamlit app demonstrates some basic image processing techniques using OpenCV.
You can upload an image and then choose an operation from the sidebar:
- **Original**: Shows the uploaded image.
- **Grayscale**: Converts the image to grayscale.
- **Threshold**: Applies a simple binary threshold to the grayscale image.
- **Canny Edge Detection**: Detects edges in the image.
- **Access Individual Pixels**: Displays a user-selected region from the image.
- **Resize the Image**: Resizes the image to a new width.
- **Rotate Image**: Rotates the image by a specified angle.
""")

# Image uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image using PIL and convert to RGB
    pil_image = Image.open(uploaded_file).convert("RGB")
    image = np.array(pil_image)

    # Display the original image
    st.image(image, caption="Original Image", use_container_width=True)
    st.write("---")
    # Sidebar for selecting the image processing operation
    operation = st.selectbox(
        "Select an operation", 
        ["Original", "Grayscale", "Threshold", "Canny Edge Detection", "Access Individual Pixels", "Resize the Image", "Rotate Image"],
        key = "BasicsSidebar")

    # Process based on the selected operation
    if operation == "Original":
        st.write("**Original Operation**: Displays the uploaded image as is.")
        st.image(image, caption="Original Image", use_container_width=True)
        with st.expander("View Code"):
            st.code("""# Simply display the original image
st.image(image, caption="Original Image", use_container_width=True)""", language="python")

    elif operation == "Grayscale":
        st.write("**Grayscale Operation**: Converts the image to shades of gray, removing color information.")
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        st.image(gray_image, caption="Grayscale Image", use_container_width=True, clamp=True)
        with st.expander("View Code"):
            st.code("""# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
st.image(gray_image, caption="Grayscale Image", use_container_width=True, clamp=True)""", language="python")

    elif operation == "Threshold":
        st.write("**Threshold Operation**: Converts the grayscale image to a binary image by applying a threshold value.")
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Apply a binary threshold (you can adjust the threshold value if needed)
        _, thresh_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        st.image(thresh_image, caption="Thresholded Image", use_container_width=True, clamp=True)
        with st.expander("View Code"):
            st.code("""# Convert to grayscale and apply binary thresholding
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
_, thresh_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
st.image(thresh_image, caption="Thresholded Image", use_container_width=True, clamp=True)""", language="python")

    elif operation == "Canny Edge Detection":
        st.write("**Canny Edge Detection Operation**: Detects the edges in the image using the Canny algorithm.")
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_image, 100, 200)
        st.image(edges, caption="Edge Detection (Canny)", use_container_width=True, clamp=True)
        with st.expander("View Code"):
            st.code("""# Convert to grayscale and apply Canny edge detection
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray_image, 100, 200)
st.image(edges, caption="Edge Detection (Canny)", use_container_width=True, clamp=True)""", language="python")

    elif operation == "Access Individual Pixels":
        st.write("**Access Individual Pixels Operation**: Extracts and displays a user-selected region from the image based on row and column values.")
        try:
            total_rows = image.shape[0]
            total_cols = image.shape[1]
            st.sidebar.write(f"Select Rows Between: 0 - {total_rows}")
            st.sidebar.write(f"Select Columns Between: 0 - {total_cols}")
            st.sidebar.warning("Please select proper range values of Rows & Columns else you will receive an Empty Image Error")
            st.sidebar.write("---")
            min_row_value = st.sidebar.number_input("Min Row Value", 0, total_rows-1, 100, 1)
            max_row_value = st.sidebar.number_input("Max Row Value", 0, total_rows-1, 200, 1)
            st.sidebar.write("---")
            min_col_value = st.sidebar.number_input("Min Column Value", 0, total_cols-1, 50, 1)
            max_col_value = st.sidebar.number_input("Max Column Value", 0, total_cols-1, 100, 1)
            
            region_of_interest = image[min_row_value : max_row_value, min_col_value : max_col_value]
            st.image(cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2RGB), caption="Selective Pixels", use_container_width=True)
            with st.expander("View Code"):
                st.code(f"""# Extract a region of interest based on user input
                total_rows, total_cols = image.shape[:2]
                min_row_value = {min_row_value}  # User-selected minimum row
                max_row_value = {max_row_value}  # User-selected maximum row
                min_col_value = {min_col_value}  # User-selected minimum column
                max_col_value = {max_col_value}  # User-selected maximum column
                region_of_interest = image[min_row_value:max_row_value, min_col_value:max_col_value]
                st.image(cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2RGB), caption="Selective Pixels", use_container_width=True)""", language="python")
        except cv2.error as e:
            st.error(f"OpenCV Error: {e}")

    elif operation == "Resize the Image":
        st.write("**Resize Operation**: Resizes the image to a new width while maintaining aspect ratio.")
        st.sidebar.write("Please enter the new width for the image.")
        new_width = st.sidebar.slider("New Width", 10, image.shape[1], 10)
        resized_image = imutils.resize(image, width=new_width)
        st.image(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB), caption="Resized Image", use_container_width=True)
        with st.expander("View Code"):
            st.code("""# Resize the image using imutils while maintaining aspect ratio
new_width = st.sidebar.slider("New Width", 10, image.shape[1], 10)
resized_image = imutils.resize(image, width=new_width)
st.image(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB), caption="Resized Image", use_container_width=True)""", language="python")

    elif operation == "Rotate Image":
        st.write("**Rotate Operation**: Rotates the image by the specified angle while ensuring the whole image fits.")
        st.sidebar.write("Please enter the angle to rotate the image.")
        angle = st.sidebar.slider("Angle", 0, 360, 0)
        rotated_image = imutils.rotate_bound(image, angle)
        st.image(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB), caption="Rotated Image", use_container_width=True)
        with st.expander("View Code"):
            st.code("""# Rotate the image using imutils
angle = st.sidebar.slider("Angle", 0, 360, 0)
rotated_image = imutils.rotate_bound(image, angle)
st.image(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB), caption="Rotated Image", use_container_width=True)""", language="python")
    
    else:
        st.write("Select an operation from the sidebar to see its effect on the image.")
else:
    st.write("Please upload an image to get started.")
