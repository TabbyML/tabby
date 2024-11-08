#!/bin/bash

# Input variables
TABBY_VERSION=${TABBY_VERSION:-dev}
LLAMA_CPP_VERSION=${LLAMA_CPP_VERSION:-b3571}
LLAMA_CPP_PLATFORM=${LLAMA_CPP_PLATFORM:-cuda-cu11.7.1-x64}
OUTPUT_NAME=${OUTPUT_NAME:-tabby_${TABBY_VERSION}_x86_64-windows-msvc-cuda117}

NAME=llama-${LLAMA_CPP_VERSION}-bin-win-${LLAMA_CPP_PLATFORM}
ZIP_FILE=${NAME}.zip

curl https://github.com/ggerganov/llama.cpp/releases/download/${LLAMA_CPP_VERSION}/${ZIP_FILE} -L -o ${ZIP_FILE}
unzip ${ZIP_FILE} -d ${OUTPUT_NAME}

pushd ${OUTPUT_NAME}
rm $(ls *.exe | grep -v "llama-server")
cp ../tabby_x86_64-windows-msvc.exe/tabby_x86_64-windows-msvc.exe tabby.exe
popd

zip -r ${OUTPUT_NAME}.zip ${OUTPUT_NAME}
rm -rf ${OUTPUT_NAME}

mkdir -p dist
mv ${OUTPUT_NAME}.zip dist/