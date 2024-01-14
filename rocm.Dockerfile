# Ubuntu version to be used as base
ARG UBUNTU_VERSION=jammy
# URL to the amdgpu-install debian package
ARG AMDGPU_INSTALL_URL=https://repo.radeon.com/amdgpu-install/6.0/ubuntu/${UBUNTU_VERSION}/amdgpu-install_6.0.60000-1_all.deb

FROM ubuntu:${UBUNTU_VERSION} as hipblas_base

ARG AMDGPU_INSTALL_URL
ARG AMDGPU_TARGETS
ENV PATH="/opt/rocm/bin:${PATH}"

# Install ROCm
RUN apt-get update &&  \
    apt-get install -y curl ca-certificates &&  \
    curl -Lo /tmp/amdgpu-install.deb "${AMDGPU_INSTALL_URL}" && \
    apt-get install -y /tmp/amdgpu-install.deb && \
    rm /tmp/amdgpu-install.deb && \
    apt-get update && \
    apt-get install -y "hipblas" && \
    apt-get purge -y curl && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

FROM hipblas_base as build

# Rust toolchain version
ARG RUST_TOOLCHAIN=stable

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        pkg-config \
        libssl-dev \
        protobuf-compiler \
        git \
        cmake \
        hipblas-dev \
        build-essential \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# setup rust.
RUN curl https://sh.rustup.rs -sSf | sh -s -- --profile minimal --no-modify-path --default-toolchain ${RUST_TOOLCHAIN} -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /root/workspace

RUN mkdir -p /opt/tabby/bin
RUN mkdir -p target

COPY . .

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/root/workspace/target \
    cargo build --features rocm --release --package tabby && \
    cp target/release/tabby /opt/tabby/bin/

FROM hipblas_base as runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Disable safe directory in docker
# Context: https://github.com/git/git/commit/8959555cee7ec045958f9b6dd62e541affb7e7d9
RUN git config --system --add safe.directory "*"

COPY --from=build /opt/tabby /opt/tabby

ENV TABBY_ROOT=/data

ENTRYPOINT ["/opt/tabby/bin/tabby"]
