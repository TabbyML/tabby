#! /bin/bash

set -e
set -x

# See https://github.com/oneapi-src/oneapi-ci for installer URLs
curl -L -nv -o webimage.exe https://registrationcenter-download.intel.com/akdlm/irc_nas/18195/w_BaseKit_p_2021.4.0.3421_offline.exe
./webimage.exe -s -x -f webimage_extracted --log extract.log
rm webimage.exe
./webimage_extracted/bootstrapper.exe -s --action install --components="intel.oneapi.win.mkl.devel:intel.oneapi.win.dnnl" --eula=accept -p=NEED_VS2017_INTEGRATION=0 -p=NEED_VS2019_INTEGRATION=0 --log-dir=.

mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$CTRANSLATE2_ROOT -DCMAKE_PREFIX_PATH="C:/Program Files (x86)/Intel/oneAPI/compiler/latest/windows/compiler/lib/intel64_win" -DBUILD_CLI=OFF -DWITH_DNNL=ON ..
cmake --build . --config Release --target install --parallel 2
cd ..
rm -r build

cp README.md python/
cp $CTRANSLATE2_ROOT/bin/ctranslate2.dll python/ctranslate2/
cp "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/windows/redist/intel64_win/compiler/libiomp5md.dll" python/ctranslate2/
cp "C:/Program Files (x86)/Intel/oneAPI/dnnl/latest/cpu_iomp/bin/dnnl.dll" python/ctranslate2/
