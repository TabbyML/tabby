# syntax = docker/dockerfile:experimental

FROM tabby_base as builder

WORKDIR /root/workspace
COPY . .

RUN mkdir -p /opt/tabby/bin
RUN mkdir -p /opt/tabby/lib
RUN mkdir -p target

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/root/workspace/target \
    cargo build --features link_shared --release && \
    cp target/release/tabby /opt/tabby/bin/

FROM ghcr.io/opennmt/ctranslate2:3.17.1-ubuntu20.04-cuda11.2

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Make link to libnvidia-ml.so (NVML) library
# so that we could get GPU stats.
RUN ln -s /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 \
        /usr/lib/x86_64-linux-gnu/libnvidia-ml.so

COPY --from=builder /opt/tabby /opt/tabby

ENV TABBY_ROOT=/data

ENTRYPOINT ["/opt/tabby/bin/tabby"]
