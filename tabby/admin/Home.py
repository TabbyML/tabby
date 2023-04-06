import streamlit as st
from utils.service_info import ServiceInfo

SERVICES = [
    ServiceInfo(label="triton", health_url="http://localhost:8002/metrics"),
    ServiceInfo(label="vector", health_url="http://localhost:8686/health"),
    ServiceInfo(label="dagu", health_url="http://localhost:8083"),
    ServiceInfo(label="server", health_url="http://localhost:8081"),
]


def make_badge_markdown(x: ServiceInfo):
    return f"![{x.label}]({x.badge_url})"


st.set_page_config(page_title="Tabby Admin - Home", layout="wide")

badges = " ".join(map(make_badge_markdown, SERVICES))
st.markdown(
    """
## Tabby
{badges}
---

**Congrats, your server is live!**

you can now query the server using `/v1/completions` endpoint:

```bash
curl -X POST http://localhost:5000/v1/completions -H 'Content-Type: application/json' --data '{
  "prompt": "def binarySearch(arr, left, right, x):\\n    mid = (left +"
}'
```

""".replace(
        "{badges}", badges
    )
)
