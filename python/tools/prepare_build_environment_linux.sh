#! /bin/bash

set -e
set -x

# Install CUDA 11.2, see:
# * https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.2.2/centos7-x86_64/base/Dockerfile
# * https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.2.2/centos7-x86_64/devel/Dockerfile
yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
yum install --setopt=obsoletes=0 -y \
    cuda-nvcc-11-2-11.2.152-1 \
    cuda-cudart-devel-11-2-11.2.152-1 \
    libcurand-devel-11-2-10.2.3.152-1 \
    libcublas-devel-11-2-11.4.1.1043-1
ln -s cuda-11.2 /usr/local/cuda

ONEAPI_VERSION=2021.4.0
MKL_BUILD=640
yum-config-manager --add-repo https://yum.repos.intel.com/oneapi
rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
yum install -y intel-oneapi-mkl-devel-$ONEAPI_VERSION-$MKL_BUILD
echo "/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin" > /etc/ld.so.conf.d/libiomp5.conf
ldconfig

pip install "cmake==3.18.4"

ONEDNN_VERSION=2.4.4
curl -L -O https://github.com/oneapi-src/oneDNN/archive/refs/tags/v${ONEDNN_VERSION}.tar.gz
tar xf *.tar.gz && rm *.tar.gz
cd oneDNN-*
cmake -DCMAKE_BUILD_TYPE=Release -DDNNL_LIBRARY_TYPE=STATIC -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_TESTS=OFF -DDNNL_ENABLE_WORKLOAD=INFERENCE -DDNNL_ENABLE_PRIMITIVE= .
make -j$(nproc) install
cd ..
rm -r oneDNN-*

mkdir build-release && cd build-release
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_CLI=OFF -DWITH_DNNL=ON -DOPENMP_RUNTIME=INTEL -DWITH_CUDA=ON -DCUDA_DYNAMIC_LOADING=ON -DCUDA_NVCC_FLAGS="-Xfatbin=-compress-all" -DCUDA_ARCH_LIST="Common" ..
VERBOSE=1 make -j$(nproc) install
cd ..
rm -r build-release

cp README.md python/
