FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04 as builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-dev \
        python3-pip \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root

ENV ONEAPI_VERSION=2023.0.0
ENV MKL_BUILD=25398
RUN wget -q https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
    apt-key add *.PUB && \
    rm *.PUB && \
    echo "deb https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        intel-oneapi-mkl-devel=$ONEAPI_VERSION-$MKL_BUILD \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip --no-cache-dir install cmake==3.22.*

ENV ONEDNN_VERSION=3.0.1
RUN wget -q https://github.com/oneapi-src/oneDNN/archive/refs/tags/v${ONEDNN_VERSION}.tar.gz && \
    tar xf *.tar.gz && \
    rm *.tar.gz && \
    cd oneDNN-* && \
    cmake -DCMAKE_BUILD_TYPE=Release -DDNNL_LIBRARY_TYPE=STATIC -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_TESTS=OFF -DDNNL_ENABLE_WORKLOAD=INFERENCE -DDNNL_ENABLE_PRIMITIVE="CONVOLUTION;REORDER" . && \
    make -j$(nproc) install && \
    cd .. && \
    rm -r oneDNN-*

COPY third_party third_party
COPY cli cli
COPY include include
COPY src src
COPY CMakeLists.txt .

ARG CXX_FLAGS
ENV CXX_FLAGS=${CXX_FLAGS:-"-msse4.1"}
ARG CUDA_NVCC_FLAGS
ENV CUDA_NVCC_FLAGS=${CUDA_NVCC_FLAGS:-"-Xfatbin=-compress-all"}
ARG CUDA_ARCH_LIST
ENV CUDA_ARCH_LIST=${CUDA_ARCH_LIST:-"Common"}
ENV CTRANSLATE2_ROOT=/opt/ctranslate2

RUN mkdir build && \
    cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=${CTRANSLATE2_ROOT} \
          -DWITH_CUDA=ON -DWITH_CUDNN=ON -DWITH_MKL=ON -DWITH_DNNL=ON -DOPENMP_RUNTIME=COMP \
          -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
          -DCUDA_NVCC_FLAGS="${CUDA_NVCC_FLAGS}" -DCUDA_ARCH_LIST="${CUDA_ARCH_LIST}" .. && \
    VERBOSE=1 make -j$(nproc) install

ENV LANG=en_US.UTF-8
COPY README.md .
COPY python python

RUN cd python && \
    python3 -m pip --no-cache-dir install -r install_requirements.txt && \
    python3 setup.py bdist_wheel --dist-dir $CTRANSLATE2_ROOT

FROM nvidia/cuda:11.2.2-base-ubuntu20.04

# We remove the cuda-compat package because it conflicts with the CUDA Enhanced Compatibility.
# See e.g. https://github.com/NVIDIA/nvidia-docker/issues/1515
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libcublas-11-2 \
        libcudnn8=8.1.1.33-1+cuda11.2 \
        libgomp1 \
        python3-pip \
        && \
    apt-get purge -y cuda-compat-11-2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV CTRANSLATE2_ROOT=/opt/ctranslate2
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CTRANSLATE2_ROOT/lib

COPY --from=builder $CTRANSLATE2_ROOT $CTRANSLATE2_ROOT
RUN python3 -m pip --no-cache-dir install $CTRANSLATE2_ROOT/*.whl && \
    rm $CTRANSLATE2_ROOT/*.whl

ENTRYPOINT ["/opt/ctranslate2/bin/ct2-translator"]
