#! /bin/bash

set -e
set -x

ROOT_DIR=$PWD
PYBIND11_VERSION=${PYBIND11_VERSION:-2.4.3}
MKL_VERSION=${MKL_VERSION:-2020.3-111}
DNNL_VERSION=${DNNL_VERSION:-1.5}
PATH=/opt/python/cp37-cp37m/bin:$PATH

yum install -y yum-utils
yum-config-manager --add-repo https://yum.repos.intel.com/mkl/setup/intel-mkl.repo
rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
yum install -y intel-mkl-64bit-${MKL_VERSION}

pip install "cmake==3.13.*"

curl -L -O https://github.com/oneapi-src/oneDNN/archive/v${DNNL_VERSION}.tar.gz
tar xf *.tar.gz && rm *.tar.gz
cd oneDNN-*
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$ROOT_DIR/dnnl -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_TESTS=OFF ..
make -j2 install
cd $ROOT_DIR

mkdir build-release && cd build-release
cmake -DCMAKE_BUILD_TYPE=Release -DLIB_ONLY=ON -DWITH_DNNL=ON -DOPENMP_RUNTIME=COMP -DCMAKE_PREFIX_PATH=$ROOT_DIR/dnnl ..
make -j2 install
cd ..
rm -r build-release

cd python
for PYTHON_ROOT in /opt/python/*
do
    $PYTHON_ROOT/bin/pip install pybind11==${PYBIND11_VERSION}
    $PYTHON_ROOT/bin/python setup.py bdist_wheel
    rm -rf build
done

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROOT_DIR/dnnl/lib64
for wheel in dist/*
do
    auditwheel show $wheel
    auditwheel repair $wheel
done
