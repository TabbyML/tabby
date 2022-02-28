#! /bin/bash

set -e
set -x

# Install test rquirements
pip install -r python/tests/requirements.txt
pip uninstall -y ctranslate2

# Download test data
curl -o transliteration-aren-all.tar.gz https://opennmt-models.s3.amazonaws.com/transliteration-aren-all.tar.gz
tar xf transliteration-aren-all.tar.gz -C tests/data/models/

curl -O https://object.pouta.csc.fi/OPUS-MT-models/en-de/opus-2020-02-26.zip
unzip opus-2020-02-26.zip -d tests/data/models/opus-mt-ende
