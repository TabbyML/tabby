import os
import jsonlines
import streamlit as st

LANGUAGE_LIST = [
    "python",
    "rust",
    "go",
    "java",
    "javascript_typescript",
    "lua",
    "php"
]

st.title(":wave: Prompt rewriting dashboard")

st.divider()
st.subheader("Select your options")

entry_count = st.slider("How many entries to view", 0, 100, 10)
language = st.radio("Select the language you are working on", LANGUAGE_LIST)

events_path = os.path.expanduser("~/.tabby/events")
log_file_name = sorted(os.listdir(events_path))[-1]
log_file_path = os.path.join(events_path, log_file_name)

prompts = []
with jsonlines.open(log_file_path) as log:
    for obj in log:
        if "completion" not in obj["event"]:
            continue
        if obj["event"]["completion"]["language"] != language:
            continue
        prompts.append(obj["event"]["completion"]["prompt"])

prompts = prompts[-entry_count:]
code_language = language if language != "javascript_typescript" else "javascript"
for i in range(len(prompts)):
    st.divider()
    prompt = prompts[i]
    st.write(f"**[prompt {i+1}]**")
    st.code(prompt, language=code_language)