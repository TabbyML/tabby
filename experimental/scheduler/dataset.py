import pandas as pd
import streamlit as st

# force wide mode
st.set_page_config(layout="wide")

st.write("Files")

# read dataframe.
df = pd.read_json("~/.tabby/dataset/data.jsonl", lines = True)

# remove useless columns
del df["git_url"]

# filter df
df = df[df["max_line_length"] < 200]
df = df[df.apply(lambda x: len(x['tags']) > 0, axis=1)]

selected = st.selectbox(
   "Filename",
   df.filepath,
)

selected_row = df[df.filepath == selected].iloc[0]

def get_range(lst, x):
    return lst[x['start']:x['end']]

if selected_row is not None:
    kinds = set([x['syntax_type_name'] for x in selected_row.tags])
    enabled_kinds = st.multiselect("Displayed Kinds", kinds, default=kinds, key=selected_row.filepath)
    col1, col2 = st.columns(2)
    
    content = selected_row.content
    with col1:
        st.write(f"File: {selected_row.filepath}")
        st.code(content, line_numbers=True)

    with col2:
        for tag in selected_row.tags:
            name = get_range(content, tag['name_range'])
            kind = tag['syntax_type_name']
            if kind not in enabled_kinds:
                continue
            is_definition = '✅' if tag['is_definition'] else '❌'
            st.markdown(f"### `{name}`\nkind: {kind}, is_definition: {is_definition}")
            st.code(get_range(content, tag['range']))