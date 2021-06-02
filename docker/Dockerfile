FROM nvidia/cuda:11.2.2-devel-ubuntu20.04 as builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-dev \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root

ENV CMAKE_VERSION=3.19.7
RUN wget -q https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION-Linux-x86_64.tar.gz && \
    tar xf *.tar.gz && \
    rm *.tar.gz
ENV PATH=$PATH:/root/cmake-$CMAKE_VERSION-Linux-x86_64/bin

ENV ONEAPI_VERSION=2021.2.0
ENV MKL_BUILD=296
ENV DNNL_BUILD=228
RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
    apt-key add *.PUB && \
    rm *.PUB && \
    echo "deb https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        intel-oneapi-mkl-devel=$ONEAPI_VERSION-$MKL_BUILD \
        intel-oneapi-dnnl-devel=$ONEAPI_VERSION-$DNNL_BUILD \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root/ctranslate2-dev

COPY third_party third_party
COPY cli cli
COPY include include
COPY src src
COPY CMakeLists.txt .

ARG CXX_FLAGS
ENV CXX_FLAGS=${CXX_FLAGS}
ARG CUDA_NVCC_FLAGS
ENV CUDA_NVCC_FLAGS=${CUDA_NVCC_FLAGS:-"-Xfatbin=-compress-all"}
ARG CUDA_ARCH_LIST
ENV CUDA_ARCH_LIST=${CUDA_ARCH_LIST:-"Common"}
ENV CTRANSLATE2_ROOT=/root/ctranslate2

RUN mkdir build && \
    cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=${CTRANSLATE2_ROOT} \
          -DCMAKE_PREFIX_PATH=${DNNL_DIR} -DWITH_DNNL=ON -DOPENMP_RUNTIME=INTEL \
          -DWITH_CUDA=ON \
          -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
          -DCUDA_NVCC_FLAGS="${CUDA_NVCC_FLAGS}" -DCUDA_ARCH_LIST="${CUDA_ARCH_LIST}" .. && \
    VERBOSE=1 make -j$(nproc) install

ENV LANG=en_US.UTF-8
COPY README.md .
COPY python python

WORKDIR /root/ctranslate2-dev/python
RUN wget -nv https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py && \
    python3 -m pip --no-cache-dir install -r install_requirements.txt && \
    python3 setup.py bdist_wheel && \
    mv dist/* /root/ctranslate2 && \
    rm -r build dist

WORKDIR /root
RUN cp -P /opt/intel/oneapi/dnnl/latest/cpu_iomp/lib/*.so* /root/ctranslate2/lib && \
    cp -P /opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libiomp5.so /root/ctranslate2/lib && \
    cp -P /usr/local/cuda/lib64/libcublas*.so* /root/ctranslate2/lib

FROM nvidia/cuda:11.2.2-base-ubuntu20.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-distutils \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/ctranslate2 /opt/ctranslate2
RUN wget -nv https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py && \
    python3 -m pip --no-cache-dir install /opt/ctranslate2/*.whl && \
    rm /opt/ctranslate2/*.whl

WORKDIR /opt

ENV CTRANSLATE2_ROOT=/opt/ctranslate2
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CTRANSLATE2_ROOT/lib

ENTRYPOINT ["/opt/ctranslate2/bin/translate"]
