import streamlit as st
from components import monaco
from utils.service_info import ServiceInfo

SERVICES = [
    ServiceInfo(label="server", url="http://server:5000"),
    ServiceInfo(label="triton", url="http://triton:8002/metrics"),
    ServiceInfo(label="vector", url="http://vector:8686/health"),
]


def make_badge_markdown(x: ServiceInfo):
    return f"![{x.label}]({x.badge_url})"


st.set_page_config(page_title="Tabby Admin")

st.markdown("## Tabby")
st.markdown(" ".join(map(make_badge_markdown, SERVICES)))

monaco.st_monaco()
