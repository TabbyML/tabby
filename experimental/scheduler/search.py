import requests
import streamlit as st
from typing import NamedTuple

class Doc(NamedTuple):
    name: str
    body: str
    score: float

    @staticmethod
    def from_json(json: dict):
        doc = json["doc"]
        return Doc(
            name=doc["name"][0],
            body=doc["body"][0],
            score=json["score"]
        )

# force wide mode
st.set_page_config(layout="wide")

query = st.text_input("Query")

if query:
    r = requests.get("http://localhost:3000/api", params=dict(q=query))
    hits = r.json()["hits"]
    for x in hits:
        doc = Doc.from_json(x)
        st.write(doc.name + " : " + str(doc.score))
        st.code(doc.body)