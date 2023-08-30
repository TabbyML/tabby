#!/bin/bash

if [[ "$OSTYPE" == "darwin"* ]]; then
  brew install protobuf
fi

if [[ "$OSTYPE" == "linux"* ]]; then
  sudo apt-get -y install protobuf-compiler
fi
