import re
import requests
import streamlit as st
from typing import NamedTuple

# force wide mode
st.set_page_config(layout="wide")

language = st.text_input("Language", "rust")

query = st.text_area("Query", "get")
tokens = re.findall(r"\w+", query)
tokens = [x for x in tokens if x != "AND" and x != "OR" and x != "NOT"]
query = "(" + " ".join(tokens) + ")" + " "  + "AND language:" + language

if query:
    r = requests.get("http://localhost:8080/v1beta/search", params=dict(q=query))
    hits = r.json()["hits"]
    for x in hits:
        st.write(x)