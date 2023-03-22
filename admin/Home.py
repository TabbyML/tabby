import streamlit as st
from utils.service_info import ServiceInfo

SERVICES = [
    ServiceInfo(label="server", url="http://server:5000"),
    ServiceInfo(label="triton", url="http://triton:8002/metrics"),
]


def make_badge_markdown(x: ServiceInfo):
    return f"![{x.label}]({x.badge_url})"


st.markdown("## Status")
st.markdown(" ".join(map(make_badge_markdown, SERVICES)))

st.markdown("## Quality")
