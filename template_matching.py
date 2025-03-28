import streamlit as st

st.title("Template Matching in Computer Vision")

# Creating tabs
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
