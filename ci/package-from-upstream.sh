#!/bin/bash

set -e

# get current bash file directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CPP_PATH="${PROJECT_ROOT}/crates/llama-cpp-server/llama.cpp"

# Input variables
LLAMA_CPP_VERSION=${LLAMA_CPP_VERSION:-$(cd ${LLAMA_CPP_PATH} && git fetch --tags origin >/dev/null && git describe --tags --abbrev=0)}
echo "LLAMA_CPP_VERSION=${LLAMA_CPP_VERSION}"
LLAMA_CPP_PLATFORM=${LLAMA_CPP_PLATFORM:-win-cuda-12.4-x64}

NAME=llama-${LLAMA_CPP_VERSION}-bin-${LLAMA_CPP_PLATFORM}
ZIP_FILE=${NAME}.zip

OUTPUT_NAME=${OUTPUT_NAME:-tabby_x86_64-windows-msvc-cuda124}

if [[ ${LLAMA_CPP_PLATFORM} == win* ]]; then
    TABBY_BINARY=${TABBY_BINARY:-tabby_x86_64-windows-msvc.exe}
    TABBY_EXTENSION=.exe
else
    TABBY_BINARY=${TABBY_BINARY:-tabby_x86_64-manylinux_2_28}
    TABBY_EXTENSION=""
fi

curl https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_CPP_VERSION}/${ZIP_FILE} -L -o ${ZIP_FILE}
unzip ${ZIP_FILE} -d ${OUTPUT_NAME}
cp "./${TABBY_BINARY}"/${TABBY_BINARY} ${OUTPUT_NAME}/tabby${TABBY_EXTENSION}

pushd ${OUTPUT_NAME}
if [[ ${LLAMA_CPP_PLATFORM} == win* ]]; then
    rm -f $(ls *.exe | grep -v -e "tabby" -e "llama-server")

    popd
    zip -r ${OUTPUT_NAME}.zip ${OUTPUT_NAME}
else
    # upstream release linux package within build/bin directory
    mv build/bin/* .
    rm -r build

    rm -f $(ls . | grep -v -e "tabby" -e "llama-server" -e '.so$' -e "LICENSE")
    mv LICENSE LICENSE-llama-server
    chmod +x llama-server tabby

    popd
    tar -czvf ${OUTPUT_NAME}.tar.gz ${OUTPUT_NAME}
fi

rm -rf "${OUTPUT_NAME}"

mkdir -p dist
mv ${OUTPUT_NAME}.* dist/
