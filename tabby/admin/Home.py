import streamlit as st
from utils.service_info import ServiceInfo

SERVICES = [
    ServiceInfo(label="triton", health_url="http://localhost:8002/metrics"),
    ServiceInfo(label="vector", health_url="http://localhost:8686/health"),
    ServiceInfo(
        label="dagu", health_url="http://localhost:8080", url="http://localhost:8080"
    ),
    ServiceInfo(
        label="server", health_url="http://localhost:5000", url="http://localhost:5000"
    ),
]


def make_badge_markdown(x: ServiceInfo):
    badge = f"![{x.label}]({x.badge_url})"
    if x.url:
        return f"[{badge}]({x.url})"
    else:
        return badge


st.set_page_config(page_title="Tabby Admin - Home")

st.markdown("## Tabby")
st.markdown(" $~$ ".join(map(make_badge_markdown, SERVICES)))
