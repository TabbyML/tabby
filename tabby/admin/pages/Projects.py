import os
from pathlib import Path

import pandas as pd
import streamlit as st
import toml
from datasets import load_from_disk
from git import Repo

st.set_page_config(page_title="Tabby Admin - Projects")

dataset_dir = os.environ.get("DATASET_DIR", None)
git_repositories_dir = os.environ.get("GIT_REPOSITORIES_DIR", None)
config_file = os.environ.get("CONFIG_FILE", None)
config = toml.load(config_file)
projects = config.get("projects", {})


def count_by_language(dataset):
    key = "language"
    df = (
        pd.DataFrame(dataset[key], columns=[key])
        .groupby([key])
        .size()
        .to_frame("# Files")
    )
    return df


def dataset_info():
    if not Path(dataset_dir).is_dir():
        st.write("*n/a*")
        return

    dataset = load_from_disk(dataset_dir)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total files", len(dataset))

    with col2:
        st.bar_chart(count_by_language(dataset))


def project_list():
    if len(projects) <= 0:
        st.write("Your project list is empty")
        st.write(f"Edit `{config_file}` to add projects")
        return

    for k, v in projects.items():
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
