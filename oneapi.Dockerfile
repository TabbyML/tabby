ARG UBUNTU_VERSION=22.04
# This needs to generally match the container host's environment.
ARG ONEAPI_VERSION=2024.0.0
# Target the CUDA build image
ARG BASE_ONEAPI_DEV_CONTAINER="intel/oneapi-basekit:${ONEAPI_VERSION}-devel-ubuntu${UBUNTU_VERSION}"
# Target the CUDA runtime image
ARG BASE_ONEAPI_RUN_CONTAINER="intel/oneapi-runtime:${ONEAPI_VERSION}-devel-ubuntu${UBUNTU_VERSION}"

FROM ${BASE_ONEAPI_DEV_CONTAINER} as build

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
        intel-opencl-icd \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# setup rust.
RUN curl https://sh.rustup.rs -sSf | bash -s -- --default-toolchain ${RUST_TOOLCHAIN} -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN update-alternatives --install "/usr/bin/cc" "cc" "${CMPLR_ROOT}/bin/icx" 100
RUN update-alternatives --install "/usr/bin/c++" "c++" "${CMPLR_ROOT}/bin/icpx" 100
RUN update-alternatives --install "/usr/bin/ld" "ld" "${CMPLR_ROOT}/bin/compiler/ld.lld" 100
ENV RUSTFLAGS="-C link-args=-fPIC -C link-args=-fsycl -C link-args=-L${CMPLR_ROOT}/lib -C link-args=-lsycl -C link-args=-lintlc"
ENV OPENSSL_NO_VENDOR=1

WORKDIR /root/workspace

RUN mkdir -p /opt/tabby/bin
RUN mkdir -p /opt/tabby/lib
RUN mkdir -p target

COPY . .

#RUN --mount=type=cache,target=/usr/local/cargo/registry \
#    --mount=type=cache,target=/root/workspace/target \
#    cargo build --features oneapi --release --package tabby && \
#    cp target/release/tabby /opt/tabby/bin/
RUN cargo build --features oneapi --release --package tabby && \
    cp target/release/tabby /opt/tabby/bin/

FROM ${BASE_ONEAPI_RUN_CONTAINER} as runtime

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
