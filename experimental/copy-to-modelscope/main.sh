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

git clone https://oauth2:${ACCESS_TOKEN}@www.modelscope.cn/$MODEL_ID.git ms_model --depth 1 || true
git clone https://huggingface.co/$MODEL_ID hf_model --depth 1 || true

echo "Sync directory"
rsync -avh --exclude '.git' --delete hf_model/ ms_model/

echo "Create README.md"
cat <<EOF >ms_model/README.md
---
license: other
tasks:
- text-generation
---

# ${MODEL_ID}

This is an mirror of [${MODEL_ID}](https://huggingface.co/${MODEL_ID}).

[Tabby](https://github.com/TabbyML/tabby) is a self-hosted AI coding assistant, offering an open-source and on-premises alternative to GitHub Copilot. It boasts several key features:
* Self-contained, with no need for a DBMS or cloud service.
* OpenAPI interface, easy to integrate with existing infrastructure (e.g Cloud IDE).
* Supports consumer-grade GPUs.
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

push_origin() {
git lfs push origin --all
git push origin
}

set -x
pushd ms_model
git add .
git commit -m "sync with upstream" || true

while true; do
	push_origin && break
done

popd

echo "Success!"
rm -rf hf_model
rm -rf ms_model
