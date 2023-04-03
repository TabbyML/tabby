import streamlit as st
from components import monaco
from utils.service_info import ServiceInfo

SERVICES = [
    ServiceInfo(label="server", url="http://localhost:5000"),
    ServiceInfo(label="triton", url="http://triton:8002/metrics"),
]


def make_badge_markdown(x: ServiceInfo):
    return f"![{x.label}]({x.badge_url})"


st.set_page_config(page_title="Tabby Admin - Home")

st.markdown("## Tabby")
st.markdown(" ".join(map(make_badge_markdown, SERVICES)))
st.markdown("---")

SNIPPETS = {
    "Clear": "# Write some code ...",
    "Fibonacci": "def fib(n):",
    "Parse JSON": """def parse_json_lines(filename: str) -> List[Any]:
    output = []
    with open(filename, "r", encoding="utf-8") as f:
""",
    "Data ORM": """import birdchirp

from birdchirp.model.chirp import chirp
from birdchirp.db.mysql import MysqlDb

class User:
    def __init__(self, user_id):
        self.user_id = user_id
        self.db = MysqlDb()

    def get_avatar""",
}


def code_editor():
    code = ""
    cols = st.columns(len(SNIPPETS))
    for col, (k, v) in zip(cols, SNIPPETS.items()):
        with col:
            if st.button(k):
                code = v
    monaco.st_monaco(key="default", code=code)


code_editor()
