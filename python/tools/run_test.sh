#! /bin/bash

set -e
set -x

ROOT_DIR=$1

# Skip tests on Python 3.8 or greater for now
if python -c "import sys; assert sys.version_info >= (3, 8)"; then
    exit
fi

pytest -v $ROOT_DIR/python/tests/test.py
