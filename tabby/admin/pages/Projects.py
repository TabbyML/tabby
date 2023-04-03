import os
from pathlib import Path

import streamlit as st
import toml
from git import Repo

st.set_page_config(page_title="Tabby Admin - Projects")

git_repositories_dir = os.environ.get("GIT_REPOSITORIES_DIR", None)
config_file = os.environ.get("CONFIG_FILE", None)
config = toml.load(config_file)
data = config.get("projects", {})


def project_list():
    for k, v in data.items():
        st.subheader(k)
        st.write(f'Git: {v["git_url"]}')

        git_repository = Path(git_repositories_dir, k)
        if not git_repository.is_dir():
            st.write(f"Status: *Before Initialization*")
            continue

        repo = Repo(git_repository)
        sha = repo.active_branch.commit.hexsha
        st.write(f"Status: `{sha}`")


if len(data) > 0:
    project_list()
else:
    st.subheader("Your project list is empty")
    st.write(f"Edit `{config_file}` to add projects")
