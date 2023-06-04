#!/bin/sh
set -e
set -x

rm -rf build || true
mkdir build

cd build && cmake ..
make

