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
    # The outdated release RPMs will be cleaned up from the city-fan.org mirror; therefore, we must first obtain the latest version.
    base_url="https://mirror.city-fan.org/ftp/contrib/sysutils/Mirroring"
    curl_rpm=$(curl -s $base_url/ | grep -oE 'curl-[0-9\.-]+\.cf\.rhel8\.x86_64\.rpm' | sort -V -u | tail -n 1)
    libcurl_rpm=$(curl -s $base_url/ | grep -oE 'libcurl-[0-9\.-]+\.cf\.rhel8\.x86_64\.rpm' | sort -V -u | tail -n 1)
    curl -SLO $base_url/$curl_rpm
    curl -SLO $base_url/$libcurl_rpm
    rpm -Uvh $curl_rpm $libcurl_rpm

    # Disable safe directory in docker
    git config --system --add safe.directory "*"

    install_protobuf_centos
  fi

  install_mailpit
fi
