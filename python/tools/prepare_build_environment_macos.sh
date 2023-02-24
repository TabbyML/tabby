#! /bin/bash

set -e
set -x

pip install "cmake==3.18.4"

mkdir build-release && cd build-release

CMAKE_EXTRA_OPTIONS=''

if [ "$CIBW_ARCHS" == "arm64" ]; then

    CMAKE_EXTRA_OPTIONS='-DCMAKE_OSX_ARCHITECTURES=arm64 -DWITH_ACCELERATE=ON -DWITH_MKL=OFF -DOPENMP_RUNTIME=NONE'

else

    # Install OneAPI MKL
    # See https://github.com/oneapi-src/oneapi-ci for installer URLs
    ONEAPI_INSTALLER_URL=https://registrationcenter-download.intel.com/akdlm/irc_nas/19080/m_BaseKit_p_2023.0.0.25441_offline.dmg
    wget -q $ONEAPI_INSTALLER_URL
    hdiutil attach -noverify -noautofsck $(basename $ONEAPI_INSTALLER_URL)
    sudo /Volumes/$(basename $ONEAPI_INSTALLER_URL .dmg)/bootstrapper.app/Contents/MacOS/bootstrapper --silent --eula accept --components intel.oneapi.mac.mkl.devel

    ONEDNN_VERSION=3.0.1
    wget -q https://github.com/oneapi-src/oneDNN/archive/refs/tags/v${ONEDNN_VERSION}.tar.gz
    tar xf *.tar.gz && rm *.tar.gz
    cd oneDNN-*
    cmake -DCMAKE_BUILD_TYPE=Release -DDNNL_LIBRARY_TYPE=STATIC -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_TESTS=OFF -DDNNL_ENABLE_WORKLOAD=INFERENCE -DDNNL_ENABLE_PRIMITIVE="CONVOLUTION;REORDER" .
    make -j$(sysctl -n hw.physicalcpu_max) install
    cd ..
    rm -r oneDNN-*

    CMAKE_EXTRA_OPTIONS='-DWITH_DNNL=ON'

fi

cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_CLI=OFF -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON $CMAKE_EXTRA_OPTIONS ..
VERBOSE=1 make -j$(sysctl -n hw.physicalcpu_max) install
cd ..
rm -r build-release

cp README.md python/
