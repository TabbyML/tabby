#!/bin/bash
set -e

MODEL_REPLICA=${MODEL_REPLICA:-1}

if [ -d "$MODEL_NAME" ]; then
MODEL_DIR="$MODEL_NAME"
else
# Get model dir.
MODEL_DIR=$(python3 <<EOF
from huggingface_hub import snapshot_download

print(snapshot_download(repo_id='$MODEL_NAME', allow_patterns='triton/**/*', local_files_only=True))
EOF
)
fi

# Set model dir in triton config.
sed -i 's@${MODEL_DIR}@'$MODEL_DIR'@g' $MODEL_DIR/triton/fastertransformer/config.pbtxt

# SET model replica in triton config.
sed -i "s/count: 1/count: $MODEL_REPLICA/g" $MODEL_DIR/triton/fastertransformer/config.pbtxt

# Start triton server.
mpirun -n 1 \
  --allow-run-as-root /opt/tritonserver/bin/tritonserver \
  --model-repository=$MODEL_DIR/triton
