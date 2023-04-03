import os
from pathlib import Path

import streamlit as st
import toml
from datasets import load_from_disk
from git import Repo

st.set_page_config(page_title="Tabby Admin - Projects")

dataset_dir = os.environ.get("DATASET_DIR", None)
git_repositories_dir = os.environ.get("GIT_REPOSITORIES_DIR", None)
config_file = os.environ.get("CONFIG_FILE", None)
config = toml.load(config_file)


def dataset_info():
    st.subheader("Dataset")
    if not Path(dataset_dir).is_dir():
        st.write("*Not populated*")
        return

    info = load_from_disk(dataset_dir)
    st.write("Source files: ", len(info))


def project_list():
    data = config.get("projects", {})

    if len(data) <= 0:
        st.write("Your project list is empty")
        st.write(f"Edit `{config_file}` to add projects")
        return

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


dataset_info()
st.write("---")
project_list()
