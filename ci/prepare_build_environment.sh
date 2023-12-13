#!/bin/bash

if [[ "$OSTYPE" == "darwin"* ]]; then
  brew install protobuf
fi

install_protobuf_centos() {
  PB_REL="https://github.com/protocolbuffers/protobuf/releases"
  curl -LO $PB_REL/download/v3.15.8/protoc-3.15.8-linux-x86_64.zip
  unzip protoc-3.15.8-linux-x86_64.zip -d /usr
  rm protoc-3.15.8-linux-x86_64.zip
}

if [[ "$OSTYPE" == "linux"* ]]; then
  if command -v apt-get ; then
    sudo apt-get -y install protobuf-compiler libopenblas-dev
  else
    # Build from manylinux2014 container
    yum -y install openblas-devel perl-IPC-Cmd unzip curl openssl-devel

    # Disable safe directory in docker
    git config --system --add safe.directory "*"

    install_protobuf_centos
  fi
fi
