#!/bin/bash
set -e

MODEL_ID=$1
ACCESS_TOKEN=$2

usage() {
  echo "Usage: $0 <model_id> <access_token>"
  exit 1
}

if [ -z "${MODEL_ID}" ]; then
  usage
fi

git clone https://huggingface.co/$MODEL_ID hf_model
git clone https://oauth2:${ACCESS_TOKEN}@www.modelscope.cn/$MODEL_ID.git ms_model

echo "Sync directory"
rsync -a --exclude '.git' hf_model/ ms_model/

echo "Create README.md"
cat <<EOF >ms_model/README.md
---
license: other
tasks:
- text-generation
---

# ${MODEL_ID}

This is an mirror of [${MODEL_ID}](https://huggingface.co/${MODEL_ID}).
EOF

echo "Create configuration.json"
cat <<EOF >ms_model/configuration.json
{
    "framework": "pytorch",
    "task": "text-generation",
    "pipeline": {
        "type": "text-generation-pipeline"
    }
}
EOF

set -x
cd ms_model
git add .
git commit -m "sync with upstream"
git push origin

echo "Success!"
rm -rf hf_model
rm -rf ms_model
