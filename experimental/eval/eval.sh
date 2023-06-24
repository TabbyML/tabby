#!/bin/bash
set -ex

docker-compose up -d

while ! curl -X POST http://localhost:8080/v1/health; do
  echo "server not ready, waiting..."
  sleep 5
done

papermill main.ipynb ./reports.ipynb -r filepattern "./tabby/dataset/*.jsonl"

jupyter nbconvert reports.ipynb --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags remove --to html

echo done
