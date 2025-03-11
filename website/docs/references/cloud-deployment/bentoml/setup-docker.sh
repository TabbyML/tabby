#!/bin/sh
set -ex

# Install tabby
DISTRO=tabby_x86_64-manylinux2014-cuda117
curl -L https://github.com/TabbyML/tabby/releases/download/v0.14.0/$DISTRO.zip \
  -o $DISTRO.zip
unzip $DISTRO.zip

chmod a+x dist/$DISTRO/*
mv dist/$DISTRO/* /usr/local/bin/
rm $DISTRO.zip
rm -rf dist

# Install katana
curl -L https://github.com/projectdiscovery/katana/releases/download/v1.1.2/katana_1.1.2_linux_amd64.zip -o katana.zip
unzip katana.zip katana
mv katana /usr/bin/
rm katana.zip

# Install rclone
curl https://rclone.org/install.sh | bash

# Config git
git config --system --add safe.directory "*"

# Download models
su bentoml -c "TABBY_MODEL_CACHE_ROOT=/home/bentoml/tabby-models tabby download --model StarCoder-1B"
su bentoml -c "TABBY_MODEL_CACHE_ROOT=/home/bentoml/tabby-models tabby download --model Qwen2-1.5B-Instruct"
su bentoml -c "TABBY_MODEL_CACHE_ROOT=/home/bentoml/tabby-models tabby download --model Nomic-Embed-Text"