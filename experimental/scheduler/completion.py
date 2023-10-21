import re
import requests
import streamlit as st
from typing import NamedTuple

# force wide mode
st.set_page_config(layout="wide")

language = st.text_input("Language", "rust")

query = st.text_area("Query", "to_owned")

if query:
    r = requests.post("http://localhost:8080/v1/completions", json=dict(segments=dict(prefix=query), language=language, debug_options=dict(return_snippets=True, return_prompt=True)))
    json = r.json()
    debug = json["debug_data"]
    snippets = debug.get("snippets", [])

    st.write("Prompt")
    st.code(debug["prompt"])

    st.write("Completion")
    st.code(json["choices"][0]["text"])

    for x in snippets:
        st.write(f"**{x['filepath']}**: {x['score']}")
        st.write(f"Length: {len(x['body'])}")
        st.code(x['body'])
