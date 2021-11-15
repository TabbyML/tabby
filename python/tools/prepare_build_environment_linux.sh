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
    libcublas-devel-11-2-11.4.1.1043-1
ln -s cuda-11.2 /usr/local/cuda

ONEAPI_VERSION=2021.4.0
MKL_BUILD=640
DNNL_BUILD=467
yum-config-manager --add-repo https://yum.repos.intel.com/oneapi
rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
yum install -y intel-oneapi-mkl-devel-$ONEAPI_VERSION-$MKL_BUILD intel-oneapi-dnnl-devel-$ONEAPI_VERSION-$DNNL_BUILD
echo "/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin" > /etc/ld.so.conf.d/libiomp5.conf
echo "/opt/intel/oneapi/dnnl/latest/cpu_iomp/lib" > /etc/ld.so.conf.d/intel-dnnl.conf
ldconfig

pip install "cmake==3.18.4"

mkdir build-release && cd build-release
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_CLI=OFF -DWITH_DNNL=ON -DOPENMP_RUNTIME=INTEL -DWITH_CUDA=ON -DCUDA_DYNAMIC_LOADING=ON -DCUDA_NVCC_FLAGS="-Xfatbin=-compress-all" -DCUDA_ARCH_LIST="Common" ..
make -j$(nproc) install
cd ..
rm -r build-release

cp README.md python/
