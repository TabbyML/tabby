#!/bin/bash

set -e

if [[ "$OSTYPE" == "darwin"* ]]; then
  brew install protobuf
fi

install_protobuf_centos() {
  PB_REL="https://github.com/protocolbuffers/protobuf/releases"
  curl -SLO $PB_REL/download/v3.15.8/protoc-3.15.8-linux-x86_64.zip
  unzip protoc-3.15.8-linux-x86_64.zip -d /usr
  rm protoc-3.15.8-linux-x86_64.zip
}

install_hipblas_5_7_2_centos() {
  curl -SL https://repo.radeon.com/amdgpu-install/5.7.2/rhel/7.9/amdgpu-install-5.7.50702-1.el7.noarch.rpm --output /tmp/amdgpu-install.rpm
  yum -y install /tmp/amdgpu-install.rpm
  rm /tmp/amdgpu-install.rpm

  yum -y install hipblas-devel hipblaslt-devel
}

if [[ "$OSTYPE" == "linux"* ]]; then
  if command -v apt-get ; then
    sudo apt-get -y install protobuf-compiler libopenblas-dev
  else
    # Build from manylinux2014 container
    yum -y install openblas-devel perl-IPC-Cmd unzip curl openssl-devel

    if [[ "$ROCM" == "5.7.2" ]]; then
      install_hipblas_5_7_2_centos
    fi

    # Disable safe directory in docker
    git config --system --add safe.directory "*"

    install_protobuf_centos
  fi
fi
