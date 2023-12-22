ARG UBUNTU_VERSION=22.04
# This needs to generally match the container host's environment.
ARG CUDA_VERSION=11.7.1
# Target the CUDA build image
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
# Target the CUDA runtime image
ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

FROM ${BASE_CUDA_DEV_CONTAINER} as build

# Rust toolchain version
ARG RUST_TOOLCHAIN=stable

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        pkg-config \
        libssl-dev \
        protobuf-compiler \
        git \
        cmake \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# setup rust.
RUN curl https://sh.rustup.rs -sSf | bash -s -- --default-toolchain ${RUST_TOOLCHAIN} -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /root/workspace

RUN mkdir -p /opt/tabby/bin
RUN mkdir -p /opt/tabby/lib
RUN mkdir -p target

COPY . .

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/root/workspace/target \
    cargo build --features cuda --release --package tabby && \
    cp target/release/tabby /opt/tabby/bin/

FROM ${BASE_CUDA_RUN_CONTAINER} as runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        openssh-client \
        ca-certificates \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Disable safe directory in docker
# Context: https://github.com/git/git/commit/8959555cee7ec045958f9b6dd62e541affb7e7d9
RUN git config --system --add safe.directory "*"

# Automatic platform ARGs in the global scope
# https://docs.docker.com/engine/reference/builder/#automatic-platform-args-in-the-global-scope
ARG TARGETARCH

# AMD64 only:
# Make link to libnvidia-ml.so (NVML) library
# so that we could get GPU stats.
RUN if [ "$TARGETARCH" = "amd64" ]; then  \
    ln -s /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 \
        /usr/lib/x86_64-linux-gnu/libnvidia-ml.so; \
    fi

COPY --from=build /opt/tabby /opt/tabby

ENV TABBY_ROOT=/data

ENTRYPOINT ["/opt/tabby/bin/tabby"]
