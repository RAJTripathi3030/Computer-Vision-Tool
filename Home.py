import streamlit as st 

basics_page = st.Page("Basics.py", title = "Basics of Computer Vision", icon = ":material/book:")
cd_page = st.Page("Color_Detection.py", title = "Color Detection", icon = ":material/palette:")
morph_page = st.Page("Morphological_Operations.py", title = "Morphological Operations", icon = ":material/transform:")
bg_sub_page = st.Page("BG_Subtraction.py", title = "Background Subtraction", icon = ":material/remove:")
# Mini-Projects
mp1_page = st.Page("Mini Projects/doc_scanner.py", title = "Document Scanner", icon = ":material/docs:")
# YOLOV8 Projects
obj_det_page = st.Page("YOLO_Obj_Detector.py", title = "Object Detection", icon = ":material/directions_car:")
obj_seg_page = st.Page("YOLO_Obj_Seg.py", title = "Object Segmentation", icon = ":material/blur_on:")
obj_tracker_page = st.Page("YOLO_Obj_Tracking.py", title = "Object Tracking", icon = ":material/track_changes:")
pos_det_page = st.Page("YOLO_Posture_Detector.py", title = "Posture Detection", icon = ":material/directions_run:")
# tester_page = st.Page("tester.py", title = "Tester", icon = ":material/tactic:")

pg = st.navigation(
    {
        "Learning" : [basics_page, cd_page, morph_page, bg_sub_page],
        "Mini Projects" : [mp1_page],
        "YOLOV8 Projects" : [obj_det_page, obj_seg_page, obj_tracker_page, pos_det_page],
    })
st.set_page_config(page_title="Computer Vision App", page_icon=":material/radio_button_unchecked:")
pg.run()