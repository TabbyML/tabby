#! /bin/bash

set -e
set -x

UNAME="$(uname -s)"
case "${UNAME}" in
    Linux*)     MACHINE=linux;;
    Darwin*)    MACHINE=macos;;
    *)          exit 1;;
esac

rm -rf build
mkdir build && cd build

if [[ "$MACHINE" == "macos" ]]; then
CMAKE_EXTRA_OPTIONS='-DCMAKE_OSX_ARCHITECTURES=arm64 -DWITH_ACCELERATE=ON -DWITH_MKL=OFF -DOPENMP_RUNTIME=NONE -DWITH_RUY=ON'
elif [[ "$MACHINE" == "linux" ]]; then
CMAKE_EXTRA_OPTIONS='-DWITH_CUDA=ON -DWITH_CUDNN=ON -DWITH_MKL=ON -DWITH_DNNL=ON -DOPENMP_RUNTIME=COMP -DCUDA_NVCC_FLAGS=-Xfatbin=-compress-all -DCUDA_ARCH_LIST=Common -DCXXFLAGS=-msse4.1'
fi


cmake -DBULID_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DBUILD_CLI=OFF -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON $CMAKE_EXTRA_OPTIONS ..

"$@"
