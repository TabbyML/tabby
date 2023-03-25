#!/bin/bash

MODEL_DIR=$(python <<EOF
from huggingface_hub import snapshot_download

print(snapshot_download(repo_id=$MODEL_NAME, allow_patterns="triton/**/*", local_files_only=True)
EOF
)
mpirun -n 1 \
  --allow-run-as-root /opt/tritonserver/bin/tritonserver \
  --model-repository=$MODEL_DIR/triton
