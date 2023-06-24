#!/bin/bash
set -ex

mkdir -p tabby
cp config.toml tabby/

docker-compose down

if nvidia-smi; then
  docker-compose -f docker-compose.yaml -f docker-compose.cuda.yaml up -d
else
  docker-compose up -d
fi

while ! curl -X POST http://localhost:8080/v1/health; do
  echo "server not ready, waiting..."
  sleep 5
done

papermill main.ipynb ./reports.ipynb -r filepattern "./tabby/dataset/*.jsonl"

jupyter nbconvert reports.ipynb --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags remove --to html

docker-compose down

echo done
