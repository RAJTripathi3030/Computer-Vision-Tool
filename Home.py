import streamlit as st 

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