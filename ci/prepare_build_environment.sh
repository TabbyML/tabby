#!/bin/bash

if [[ "$OSTYPE" == "darwin"* ]]; then
  brew install protobuf
fi

if [[ "$OSTYPE" == "linux"* ]]; then
  apt-get -y install protobuf
fi
