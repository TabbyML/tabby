FROM ghcr.io/opennmt/ctranslate2:3.15.0-ubuntu20.04-cuda11.2 as source
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04 as builder

ENV CTRANSLATE2_ROOT=/opt/ctranslate2
COPY --from=source $CTRANSLATE2_ROOT $CTRANSLATE2_ROOT

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        pkg-config \
        libssl-dev \
        protobuf-compiler \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# setup rust.
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /root/workspace
COPY Cargo.toml Cargo.toml
COPY Cargo.lock Cargo.lock
COPY crates crates

RUN mkdir -p /opt/tabby/bin
RUN mkdir -p /opt/tabby/lib
RUN mkdir -p target

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/root/workspace/target \
    cargo build --features link_shared --release && \
    cp target/release/tabby /opt/tabby/bin/

FROM ghcr.io/opennmt/ctranslate2:3.15.0-ubuntu20.04-cuda11.2

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/tabby /opt/tabby

ENV TABBY_ROOT=/data

ENTRYPOINT ["/opt/tabby/bin/tabby"]
