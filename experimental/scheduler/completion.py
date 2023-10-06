import re
import requests
import streamlit as st
from typing import NamedTuple

class Doc(NamedTuple):
    name: str
    body: str
    score: float
    filepath: str

    @staticmethod
    def from_json(json: dict):
        doc = json["doc"]
        return Doc(
            name=doc["name"][0],
            body=doc["body"][0],
            score=json["score"],
            filepath=doc["filepath"][0],
        )

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
        doc = Doc.from_json(x)
        st.write(doc.name + "@" + doc.filepath + " : " + str(doc.score))
        st.code(doc.body)
