FROM nvidia/cuda:12.2.2-devel-ubuntu20.04 as builder

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        cmake \
        git \
        unzip \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV ONEAPI_VERSION=2022.0.1
ENV MKL_BUILD=117
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

WORKDIR /root

ENV MARIAN_VERSION=1.11.0
RUN git clone --recursive --branch $MARIAN_VERSION --depth 1 https://github.com/marian-nmt/marian-dev.git && \
    cd marian-dev && \
    mkdir build && cd build && \
    MKLROOT=/opt/intel/oneapi/mkl/latest cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc)

ENV MOSESDECODER_VERSION=RELEASE-4.0
RUN git clone --branch $MOSESDECODER_VERSION --depth 1 https://github.com/moses-smt/mosesdecoder.git

RUN wget -q https://object.pouta.csc.fi/OPUS-MT-models/en-de/opus-2020-02-26.zip && \
    unzip *.zip -d /model && \
    rm *.zip

COPY tokenize detokenize translate /
