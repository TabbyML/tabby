import os

import streamlit as st
import toml

st.set_page_config(page_title="Tabby Admin - Projects")

config_file = os.environ.get("CONFIG_FILE", None)
config = toml.load(config_file)
data = config.get("repositories", {})


def project_list():
    for k, v in data.items():
        st.subheader(k)
        st.write(v["git_url"])


if len(data) > 0:
    project_list()
else:
    st.subheader("Your project list is empty")
    st.write(f"Edit `{config_file}` to add projects")
