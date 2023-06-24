#!/bin/bash
set -ex

if [ -z "$TABBY_ROOT"]; then
  export TABBY_ROOT="$HOME/.tabby"
fi

papermill main.ipynb ./reports.ipynb -r filepattern "$TABBY_ROOT/dataset/*.jsonl"

jupyter nbconvert reports.ipynb --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags remove --to html
