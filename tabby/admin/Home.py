import streamlit as st
from streamlit_js_eval import get_page_location
from utils.service_info import ServiceInfo
from utils.streamlit import set_page_config

SERVICES = [
    ServiceInfo(label="triton", health_url="http://localhost:8002/metrics"),
    ServiceInfo(label="vector", health_url="http://localhost:8686/health"),
    ServiceInfo(label="dagu", health_url="http://localhost:8083"),
    ServiceInfo(label="server", health_url="http://localhost:8081"),
]


def make_badge_markdown(x: ServiceInfo):
    return f"![{x.label}]({x.badge_url})"


set_page_config(page_title="Home")

badges = " ".join(map(make_badge_markdown, SERVICES))
server_url = get_page_location()["origin"]
st.markdown(
    """
## Tabby [![github star](https://img.shields.io/github/stars/TabbyML/tabby?style=social)](http://github.com/TabbyML/tabby)

{badges}
---

**Congrats, your server is live!**

Below is the URL for your API endpoint.
```
{server_url}
```

### Clients

* [VSCode](https://marketplace.visualstudio.com/items?itemName=TabbyML.vscode-tabby)

""".replace(
        "{badges}", badges
    ).replace(
        "{server_url}", server_url
    )
)
