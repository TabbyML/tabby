#! /bin/bash

set -e
set -x

ROOT_DIR=$PWD
PATH=/opt/python/cp37-cp37m/bin:$PATH

# Install CUDA 10.1, see:
# * https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/10.1/centos6-x86_64/base/Dockerfile
# * https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/10.1/centos6-x86_64/devel/Dockerfile
CUDA_VERSION=10.1.243
CUDA_PKG_VERSION=10-1-$CUDA_VERSION-1
curl -o cuda-repo.rpm https://developer.download.nvidia.com/compute/cuda/repos/rhel6/x86_64/cuda-repo-rhel6-$CUDA_VERSION-1.x86_64.rpm
rpm --install cuda-repo.rpm
yum clean expire-cache
yum install -y cuda-nvcc-$CUDA_PKG_VERSION cuda-cudart-dev-$CUDA_PKG_VERSION libcublas-devel-10.2.1.243-1
ln -s cuda-10.1 /usr/local/cuda
mv /usr/local/cuda-10.2/include/* /usr/local/cuda/include/
mv /usr/local/cuda-10.2/lib64/lib* /usr/local/cuda/lib64/

MKL_VERSION=2020.4-912
yum install -y yum-utils
yum-config-manager --add-repo https://yum.repos.intel.com/mkl/setup/intel-mkl.repo
rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
yum install -y intel-mkl-64bit-${MKL_VERSION}

pip install "cmake==3.13.*"

DNNL_VERSION=1.7
curl -L -O https://github.com/oneapi-src/oneDNN/archive/v${DNNL_VERSION}.tar.gz
tar xf *.tar.gz && rm *.tar.gz
cd oneDNN-*
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$ROOT_DIR/dnnl -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_TESTS=OFF ..
make -j2 install
cd $ROOT_DIR

mkdir build-release && cd build-release
cmake -DCMAKE_BUILD_TYPE=Release -DLIB_ONLY=ON -DWITH_DNNL=ON -DOPENMP_RUNTIME=COMP -DCMAKE_PREFIX_PATH=$ROOT_DIR/dnnl -DWITH_CUDA=ON -DCUDA_NVCC_FLAGS="-Xfatbin -compress-all" -DCUDA_ARCH_LIST="Common" ..
make -j2 install
cd ..
rm -r build-release

cd python
for PYTHON_ROOT in /opt/python/*
do
    $PYTHON_ROOT/bin/pip install -r install_requirements.txt
    $PYTHON_ROOT/bin/python setup.py bdist_wheel
    rm -rf build
done

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROOT_DIR/dnnl/lib64
for wheel in dist/*
do
    auditwheel show $wheel
    auditwheel repair $wheel
done
