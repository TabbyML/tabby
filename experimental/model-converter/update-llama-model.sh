#!/bin/bash
set -e

ACCESS_TOKEN=$1

usage() {
  echo "Usage: $0 <access_token>"
  exit 1
}

if [ -z "${ACCESS_TOKEN}" ]; then
  usage
fi

prepare_llama_cpp() {
  git clone https://github.com/ggerganov/llama.cpp.git
  pushd llama.cpp

  git checkout 6961c4bd0b5176e10ab03b35394f1e9eab761792
  mkdir build
  pushd build
  cmake ..
  make quantize
  popd
  popd
}

update_model() {
  MODEL_ID=$1

  git clone https://${ACCESS_TOKEN}@huggingface.co/$MODEL_ID hf_model --depth 1

  pushd hf_model
  huggingface-cli lfs-enable-largefiles .

  python ../llama.cpp/convert-starcoder-hf-to-gguf.py  . --outfile ./ggml/f16.v2.gguf 1
  ../llama.cpp/build/bin/quantize ./ggml/f16.v2.gguf ./ggml/q8_0.v2.gguf q8_0

  git add .
  git commit -m "add ggml model v2"
  git lfs push origin
  git push origin
  popd

  echo "Success!"
  rm -rf hf_model
}

set -x
prepare_llama_cpp || true

# update_model TabbyML/StarCoder-1B
# update_model TabbyML/StarCoder-3B
update_model TabbyML/StarCoder-7B
update_model TabbyML/CodeLlama-7B
update_model TabbyML/CodeLlama-13B
update_model TabbyML/Mistral-7B
update_model TabbyML/WizardCoder-3B
