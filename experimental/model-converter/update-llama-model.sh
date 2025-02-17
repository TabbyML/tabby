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
  git clone https://github.com/ggml-org/llama.cpp.git
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
  CONVERTER=$1
  MODEL_ID=$2

  git clone https://huggingface.co/$MODEL_ID hf_model --depth 1

  pushd hf_model
  huggingface-cli lfs-enable-largefiles .

  "$CONVERTER"
  ../llama.cpp/build/bin/quantize ./ggml/f16.v2.gguf ./ggml/q8_0.v2.gguf q8_0
  huggingface-cli upload $MODEL_ID ggml/q8_0.v2.gguf ggml/q8_0.v2.gguf
  popd

  echo "Success!"
  rm -rf hf_model
}

starcoder() {
  python ../llama.cpp/convert-starcoder-hf-to-gguf.py  . --outfile ./ggml/f16.v2.gguf 1
}

llama() {
  python ../llama.cpp/convert.py  . --outfile ./ggml/f16.v2.gguf --outtype f16
}

set -x
huggingface-cli login --token ${ACCESS_TOKEN}

prepare_llama_cpp || true

update_model starcoder TabbyML/StarCoder-1B
update_model starcoder TabbyML/StarCoder-3B
update_model starcoder TabbyML/StarCoder-7B
update_model llama TabbyML/CodeLlama-7B
update_model llama TabbyML/CodeLlama-13B
update_model llama TabbyML/Mistral-7B
update_model starcoder TabbyML/WizardCoder-3B
