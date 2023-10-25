FROM nvcr.io/nvidia/l4t-cuda:11.4.19-devel as builder

ENV DEBIAN_FRONTEND=noninteractive
# Use kitware's repo to make sure CMake is newer than 3.26, which is required by CTranslate2
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null |\
        gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null && \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' |\
        tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        pkg-config \
        libssl-dev \
        libcudnn8-dev \
        protobuf-compiler \
        git \
        cmake \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# setup rust.
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /root/workspace
COPY . .

RUN mkdir -p /opt/tabby/bin
RUN mkdir -p /opt/tabby/lib
RUN mkdir -p target

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/root/workspace/target \
    cargo build --features link_static_cuda --release && \
    cp target/release/tabby /opt/tabby/bin/

FROM nvcr.io/nvidia/l4t-cuda:11.4.19-runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git libcudnn8 tini && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Disable safe directory in docker
# Context: https://github.com/git/git/commit/8959555cee7ec045958f9b6dd62e541affb7e7d9
RUN git config --system --add safe.directory "*"

COPY --from=builder /opt/tabby /opt/tabby

ENV TABBY_ROOT=/data

ENTRYPOINT ["/usr/bin/tini", "--", "/opt/tabby/bin/tabby"]
