# Prompt rewriting evaluation tool

## Install dependencies
```
pip install -r requirements.txt
```

## Run backend rewriting script
1. tweak `eval.toml`
    - tabby binary path
    - index repo url (the repo you want tabby to index from)
    - sample repo url (the repo you want to generate completion requests from)
    - language
    - prompt count

2. run `python evaluation.py`

## Run dashboard to view prompts
```
streamlit run dashboard.py
```
- Tweak the slider bar to change how many recent prompts you want to review.
- Change the language to filter only the specific language you are interested in.