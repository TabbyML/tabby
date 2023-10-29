# Start with Ubuntu 23.04 as base image
FROM rust as build

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        pkg-config \
        libssl-dev \
        protobuf-compiler \
        git \
        cmake \
        build-essential gcc make cmake git libopencl1 ocl-icd-opencl-dev clblast-utils \ 
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# setup rust.
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /root/workspace
COPY ../../ .


RUN mkdir -p /opt/tabby/bin
RUN mkdir -p /opt/tabby/lib
RUN mkdir -p target

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/root/workspace/target \
    cargo build --features clblast --release --package tabby && \
    cp target/release/tabby /opt/tabby/bin/

FROM ubuntu:23.04 as runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git curl ca-certificates \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Disable safe directory in docker
# Context: https://github.com/git/git/commit/8959555cee7ec045958f9b6dd62e541affb7e7d9
RUN git config --system --add safe.directory "*"

# # Make link to libnvidia-ml.so (NVML) library
# # so that we could get GPU stats.
# RUN ln -s /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 \
#         /usr/lib/x86_64-linux-gnu/libnvidia-ml.so

COPY --from=build /opt/tabby /opt/tabby

ENV TABBY_ROOT=/data

ENTRYPOINT ["/opt/tabby/bin/tabby"]