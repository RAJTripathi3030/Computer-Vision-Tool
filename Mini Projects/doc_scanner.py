import cv2 
import imutils 
import numpy as np 
import streamlit as st
from PIL import Image

# order_points() : 
# 1. This function takes in 4 points and re-orders them in a specific order.
# 2. The order is as follows: top-left, top-right, bottom-right, bottom-left.
# 3. Line 15 : First I create a matrix of (4,2) to store the points.
# 4. Line 21-23 : 
#     I calculate the sum of points, row wise.
#     I find the index of the smallest sum and largest sum.
#     I assign the top-left and bottom-right points to the respective indices.
# 5. Line 28-30 :
#     I calculate the difference of points, row wise.
#     I find the index of the smallest difference and largest difference.
#     I assign the top-right and bottom-left points to the respective indices.

def order_points(points):
    rect = np.zeros((4, 2), dtype = "float32" )
    
    sum_of_points = points.sum(axis = 1)
    rect[0] = points[np.argmin(sum_of_points)]
    rect[2] = points[np.argmax(sum_of_points)]
    
    difference = np.diff(points, axis = 1)
    rect[1] =  points[np.argmin(difference)]
    rect[3] = points[np.argmax(difference)]
    
    return rect


# Applies a four-point perspective transform to an image.

# This function takes an image and a set of four points defining a 
# quadrilateral region in the image. It then applies a perspective 
# transformation to warp the image so that the selected region appears 
# as a top-down, rectangular view.

# Parameters:
# -----------
# image : numpy.ndarray
#     The input image to be transformed.
# points : numpy.ndarray
#     A 4x2 NumPy array containing the (x, y) coordinates of the four 
#     points defining the region to be transformed.

# Returns:
# --------
# numpy.ndarray
#     The warped (transformed) image with a corrected perspective.

# Notes:
# ------
# - The function first orders the points in the correct order: 
#     top-left, top-right, bottom-right, and bottom-left.
# - It computes the width and height of the transformed image based on 
#     the Euclidean distances between the points.
# - A perspective transformation matrix is computed and applied using 
#     OpenCV's `cv2.getPerspectiveTransform` and `cv2.warpPerspective`.

# Example:
# --------
# >>> import cv2
# >>> import numpy as np
# >>> image = cv2.imread("image.jpg")
# >>> pts = np.array([[100, 200], [400, 200], [400, 500], [100, 500]], dtype="float32")
# >>> warped_image = four_point_perspective_transform(image, pts)
# >>> cv2.imshow("Warped", warped_image)
# >>> cv2.waitKey(0)
# >>> cv2.destroyAllWindows()

def four_point_perspective_transform(image, points):
    rect = order_points(points)
    (topLeft, topRight, bottomRight, bottomLeft) = rect
    
    widthA = np.linalg.norm(bottomRight - bottomLeft)
    widthB = np.linalg.norm(topRight - topLeft)
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.linalg.norm(topRight - bottomRight)
    heightB = np.linalg.norm(topLeft - bottomLeft)
    maxHeight = max(int(heightA), int(heightB))
    
    destination_points = np.array([
        [0,0], 
        [maxWidth - 1, 0], 
        [maxWidth - 1, maxHeight - 1], 
        [0, maxHeight - 1]], dtype = "float32")
    
    perspective_transform_matrix = cv2.getPerspectiveTransform(rect, destination_points)
    warped = cv2.warpPerspective(image, perspective_transform_matrix, (maxWidth, maxHeight))
    
    return warped

#----------------------------------------------------------MAIN SCRIPT----------------------------------------------------------------

st.title("Document Scanner")
st.write("This page demonstrates an implementation of how a document scanner works using OpenCV.")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    image = np.array(pil_image)
    st.image(image, caption="Original Image", use_container_width=True)
        
    # Making a copy of the real image
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height = 500)
        
    # Now I perform Grayscaling, Blurring and Edge Detection on the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    with st.expander("See the Grayscaled, Blurred and Edged Image"):
        st.image(gray, caption="Grayscale Image", use_container_width=True)
        st.image(edged, caption="Edged Image", use_container_width=True)
            
    # Finding contours in the edged image
    contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    
    screenCnt = None 
    
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        if len(approx) == 4:
            screenCnt = approx
            break
    
    if screenCnt is None:
        st.error("Couldn't find the document contour")
        
    points = screenCnt.reshape(4, 2) * ratio 
    
    wrapped = four_point_perspective_transform(orig, points)
    
    wrapped_gray = cv2.cvtColor(wrapped, cv2.COLOR_BGR2GRAY)
    wrapped_gray = imutils.resize(wrapped_gray, height = 500)
    # wrapped_thresh = cv2.threshold(wrapped_gray, 0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    
    if st.button("Show the final image"):
        col1, col2 = st.columns(2)
        st.balloons()
        with col1: 
            st.image(image, caption="Final Image", use_container_width=True)
        with col2:
            st.image(wrapped_gray, caption="Final Image", use_container_width=True)