#!/bin/bash

# get current bash file directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CPP_PATH="${PROJECT_ROOT}/crates/llama-cpp-server/llama.cpp"

# Input variables
LLAMA_CPP_VERSION=${LLAMA_CPP_VERSION:-$(cd ${LLAMA_CPP_PATH} && git fetch --tags origin >/dev/null && git describe --tags --abbrev=0)}
echo "LLAMA_CPP_VERSION=${LLAMA_CPP_VERSION}"
LLAMA_CPP_PLATFORM=${LLAMA_CPP_PLATFORM:-cuda-cu11.7-x64}
OUTPUT_NAME=${OUTPUT_NAME:-tabby_x86_64-windows-msvc-cuda117}

NAME=llama-${LLAMA_CPP_VERSION}-bin-win-${LLAMA_CPP_PLATFORM}
ZIP_FILE=${NAME}.zip

curl https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_CPP_VERSION}/${ZIP_FILE} -L -o ${ZIP_FILE}
unzip ${ZIP_FILE} -d ${OUTPUT_NAME}

pushd ${OUTPUT_NAME}
rm $(ls *.exe | grep -v "llama-server")
cp ../tabby_x86_64-windows-msvc.exe/tabby_x86_64-windows-msvc.exe tabby.exe
popd

zip -r ${OUTPUT_NAME}.zip ${OUTPUT_NAME}
rm -rf "${OUTPUT_NAME}"

mkdir -p dist
mv ${OUTPUT_NAME}.zip dist/
