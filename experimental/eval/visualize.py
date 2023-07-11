import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(layout="wide")

df = pd.read_json("reports.jsonl", lines=True)

for _, v in df.iterrows():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("prompt")
        st.code(v.prompt)
    with col2:
        st.write("prediction")
        st.code(v.prediction)
        st.write("label")
        st.code(v.label)
    with col3:
        col1, col2 = st.columns(2)
        st.metric("Line score", v.line_score)
        st.metric("Block score", v.block_score)
    st.divider()
