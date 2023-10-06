import requests
import streamlit as st
from typing import NamedTuple

# force wide mode
st.set_page_config(layout="wide")

query = st.text_input("Query")

if query:
    r = requests.get("http://localhost:8080/v1beta/search", params=dict(q=query))
    hits = r.json()["hits"]
    for x in hits:
        st.write(x)
