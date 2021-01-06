#! /bin/bash

set -e
set -x

# Install OneAPI MKL and DNNL
# See https://github.com/oneapi-src/oneapi-ci for installer URLs
ONEAPI_INSTALLER_URL=https://registrationcenter-download.intel.com/akdlm/irc_nas/17426/m_BaseKit_p_2021.1.0.2427_offline.dmg
wget -q $ONEAPI_INSTALLER_URL
hdiutil attach -noverify -noautofsck $(basename $ONEAPI_INSTALLER_URL)
sudo /Volumes/$(basename $ONEAPI_INSTALLER_URL .dmg)/bootstrapper.app/Contents/MacOS/bootstrapper --silent --eula accept --components intel.oneapi.mac.mkl.devel:intel.oneapi.mac.dnnl
# Prefix the install name of libdnnl with @rpath
sudo install_name_tool -id @rpath/libdnnl.2.dylib /opt/intel/oneapi/dnnl/latest/cpu_iomp/lib/libdnnl.2.0.dylib

# Install LLVM's libomp because Intel's OpenMP runtime included in MKL does
# not ship with the header file.
brew install libomp

pip install "cmake==3.18.4"

mkdir build-release && cd build-release
cmake -DCMAKE_BUILD_TYPE=Release -DLIB_ONLY=ON -DWITH_DNNL=ON -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_CXX_FLAGS="-Wno-unused-command-line-argument" ..
make -j$(sysctl -n hw.physicalcpu_max) install
cd ..
rm -r build-release

rm python/README.md && cp README.md python/README.md
