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
    # Build from manylinux_2_28 container

    # CentOS 7 is EOL after 2024 06, need to update to vault.centos.org
    sed -i -e 's/mirrorlist/#mirrorlist/g' -e 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-*
    yum -y install openblas-devel perl-IPC-Cmd unzip openssl-devel

    # Install newer version of curl to support `--fail-with-body` on vulkan-sdk-install
    curl -SLO https://mirror.city-fan.org/ftp/contrib/sysutils/Mirroring/curl-8.12.1-1.0.cf.rhel8.x86_64.rpm
    curl -SLO https://mirror.city-fan.org/ftp/contrib/sysutils/Mirroring/libcurl-8.12.1-1.0.cf.rhel8.x86_64.rpm
    rpm -Uvh curl-8.12.1-1.0.cf.rhel8.x86_64.rpm libcurl-8.12.1-1.0.cf.rhel8.x86_64.rpm

    # Disable safe directory in docker
    git config --system --add safe.directory "*"

    install_protobuf_centos
  fi

  install_mailpit
fi
