#! /bin/bash

set -e
set -x

# Install CUDA 10.1, see:
# * https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/10.1/centos7-x86_64/base/Dockerfile
# * https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/10.1/centos7-x86_64/devel/Dockerfile
CUDA_VERSION=10.1.243
CUDA_PKG_VERSION=10-1-$CUDA_VERSION-1
CUBLAS_PKG_VERSION=10.2.1.243-1
curl -o cuda-repo.rpm https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-repo-rhel7-$CUDA_VERSION-1.x86_64.rpm
rpm --install cuda-repo.rpm
yum clean expire-cache
yum install --setopt=obsoletes=0 -y cuda-nvcc-$CUDA_PKG_VERSION cuda-cudart-dev-$CUDA_PKG_VERSION libcublas-devel-$CUBLAS_PKG_VERSION libcublas10-$CUBLAS_PKG_VERSION
ln -s cuda-10.1 /usr/local/cuda

# Maximum GCC version supported by CUDA 10.1 is GCC 8.
yum install -y devtoolset-8
source /opt/rh/devtoolset-8/enable

ONEAPI_VERSION=2021.2.0
MKL_BUILD=296
DNNL_BUILD=228
yum install -y yum-utils
yum-config-manager --add-repo https://yum.repos.intel.com/oneapi
rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
yum install -y intel-oneapi-mkl-devel-$ONEAPI_VERSION-$MKL_BUILD intel-oneapi-dnnl-devel-$ONEAPI_VERSION-$DNNL_BUILD
echo "/opt/intel/oneapi/dnnl/latest/cpu_gomp/lib" > /etc/ld.so.conf.d/intel-dnnl.conf
ldconfig

pip install "cmake==3.18.4"

mkdir build-release && cd build-release
cmake -DCMAKE_BUILD_TYPE=Release -DLIB_ONLY=ON -DWITH_DNNL=ON -DOPENMP_RUNTIME=COMP -DWITH_CUDA=ON -DCUDA_NVCC_FLAGS="-Xfatbin -compress-all" -DCUDA_ARCH_LIST="Common" ..
make -j$(nproc) install
cd ..
rm -r build-release

/usr/bin/cp --remove-destination README.md python/README.md
