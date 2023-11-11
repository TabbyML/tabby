# Start with Ubuntu 23.10 as base image
FROM ubuntu:23.10 as build

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Update Ubuntu Software repository
RUN apt update

# Install necessary dependencies
#RUN apt install -y wget gpg curl build-essential cmake git libopencl1 ocl-icd-opencl-dev clblast-utils opencl-headers ocl-icd libclblast-dev
RUN apt-get install -y wget gpg curl build-essential cmake git opencl-headers

# Add Intel's package repository
RUN wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
    gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg && \
    echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client" | \
    tee /etc/apt/sources.list.d/intel-gpu-jammy.list && \
    apt-get update

# Install the user-mode packages
RUN apt-get install -y \
    intel-opencl-icd intel-level-zero-gpu level-zero \
    intel-media-va-driver-non-free libmfx1 libmfxgen1 libvpl2 \
    libegl-mesa0 libegl1-mesa libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri \
    libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 libxatracker2 mesa-va-drivers \
    mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all vainfo hwinfo clinfo

WORKDIR /opt


RUN apt install -y libopencl1 ocl-icd-opencl-dev clblast-utils opencl-headers libclblast-dev

# Install CLBlast
RUN git clone https://github.com/CNugteren/CLBlast.git && \
    mkdir CLBlast/build && \
    cd CLBlast/build && \
    cmake .. -DBUILD_SHARED_LIBS=OFF -DTUNERS=OFF && \
    cmake --build . --config Release && \
    cmake --install . --prefix /usr/local/ && \
    cd /opt
    
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        pkg-config \
        libssl-dev \
        protobuf-compiler \
        git \
        cmake \
        build-essential gcc make \ 
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root/workspace
COPY ../.. .


RUN mkdir -p /opt/tabby/bin
RUN mkdir -p /opt/tabby/lib
RUN mkdir -p target

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Add Cargo to the PATH
ENV PATH="/root/.cargo/bin:${PATH}"

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/root/workspace/target \
    cargo build --features opencl --release --package tabby && \
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

COPY --from=build /opt/tabby /opt/tabby

ENV TABBY_ROOT=/data

ENTRYPOINT ["/opt/tabby/bin/tabby"]