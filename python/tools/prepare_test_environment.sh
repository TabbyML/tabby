#! /bin/bash

set -e
set -x

# Skip tests on Python 3.8 or greater for now
if python -c "import sys; assert sys.version_info >= (3, 8)"; then
    exit
fi

# Install test rquirements
pip install -r python/tests/requirements.txt

# Download test data
curl -o transliteration-aren-all.tar.gz https://opennmt-models.s3.amazonaws.com/transliteration-aren-all.tar.gz
tar xf transliteration-aren-all.tar.gz -C tests/data/models/
