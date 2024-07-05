#!/bin/bash

install_protobuf_centos() {
  PB_REL="https://github.com/protocolbuffers/protobuf/releases"
  curl -LO $PB_REL/download/v3.15.8/protoc-3.15.8-linux-x86_64.zip
  unzip protoc-3.15.8-linux-x86_64.zip -d /usr
  rm protoc-3.15.8-linux-x86_64.zip
}

install_mailpit() {
  bash < <(curl -sL https://raw.githubusercontent.com/axllent/mailpit/develop/install.sh)
}

if [[ "$OSTYPE" == "darwin"* ]]; then
  brew install protobuf
  install_mailpit
fi

if [[ "$OSTYPE" == "linux"* ]]; then
  if command -v apt-get ; then
    apt-get -y install protobuf-compiler libopenblas-dev sqlite3 graphviz libcurl4-openssl-dev
  else
    # Build from manylinux2014 container

    # CentOS 7 is EOL after 2024 06, need to update to vault.centos.org
    sed -i -e 's/mirrorlist/#mirrorlist/g' -e 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-*
    yum -y install openblas-devel perl-IPC-Cmd unzip curl openssl-devel

    # Disable safe directory in docker
    git config --system --add safe.directory "*"

    install_protobuf_centos
  fi

  install_mailpit
fi
