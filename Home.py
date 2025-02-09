import streamlit as st 
from streamlit_option_menu import option_menu
import importlib


# Sidebar Navigation (Defined only in Home.py)
# with st.sidebar:
#     selected_page = option_menu(
#         menu_title="Navigation",        
#         options=["Home", "Basics", "Color Detection", "Document Scanner (Mini Project)"],   
#         icons=["house", "book", "palette", "file-pdf"],  
#         # menu_icon="cast",              
#         default_index=0,               
#         orientation="vertical",
#         key="HomeSidebar"  # Unique key to prevent duplicate ID error
#     )

# # Dynamically load the selected page
# if selected_page == "Home":
#     st.title("About This Tool")
#     st.write("This is a tool designed to aid your Computer Vision learning journey. It allows you to perform various Computer Vision tasks such as image classification, object detection, and image segmentation.")
# elif selected_page == "Basics":
#     page = importlib.import_module("Basics")
#     page.main()
# elif selected_page == "Color Detection":
#     page = importlib.import_module("Color_Detection")
#     page.main()
# elif selected_page == "Document Scanner (Mini Project)":
#     page = importlib.import_module("Document_Scanner(Mini-Project)")
#     page.main()


basics_page = st.Page("Basics.py", title = "Basics of Computer Vision", icon = ":material/book:")
cd_page = st.Page("Color_Detection.py", title = "Color Detection", icon = ":material/palette:")
mp1_page = st.Page("Mini Projects/doc_scanner.py", title = "Document Scanner", icon = ":material/docs:")
tester_page = st.Page("Tester.py", title = "Tester", icon = ":material/tactic:")

pg = st.navigation(
    {
        "Learning" : [basics_page, cd_page, tester_page],
        "Mini Projects" : [mp1_page]
    })
st.set_page_config(page_title="Computer Vision App", page_icon=":material/radio_button_unchecked:")
pg.run()