import os

import streamlit as st

hide_streamlit_menu = os.environ.get("STREAMLIT_HIDE_MENU", "True") == "True"


def set_page_config(page_title, **kwargs):
    st.set_page_config(
        page_title=f"Tabby Admin - {page_title}", layout="wide", **kwargs
    )
    if hide_streamlit_menu:
        hide_streamlit_style = (
            "<style>#MainMenu {visibility: hidden;}footer {visibility: hidden;}</style>"
        )
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)
