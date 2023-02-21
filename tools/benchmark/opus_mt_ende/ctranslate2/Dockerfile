FROM opennmt/ctranslate2:3.6.0-ubuntu20.04-cuda11.2

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        unzip \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root

# Compile spm_encode via Marian.
ENV MARIAN_VERSION=1.11.0
RUN git clone --recursive --branch $MARIAN_VERSION --depth 1 https://github.com/marian-nmt/marian-dev.git && \
    cd marian-dev && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCOMPILE_CUDA=OFF -DUSE_MKL=OFF .. && \
    make -j$(nproc) spm_encode

ENV MOSESDECODER_VERSION=RELEASE-4.0
RUN git clone --branch $MOSESDECODER_VERSION --depth 1 https://github.com/moses-smt/mosesdecoder.git

RUN wget -q https://object.pouta.csc.fi/OPUS-MT-models/en-de/opus-2020-02-26.zip && \
    unzip *.zip -d /marian-model && \
    rm *.zip && \
    ct2-opus-mt-converter --model_dir /marian-model --output_dir /model && \
    cp /marian-model/*.sh /model && \
    cp /marian-model/*.spm /model && \
    rm -r /marian-model

COPY tokenize detokenize translate /
