FROM ubuntu:16.04 as builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        cpio \
        g++-8 \
        gcc-8 \
        libboost-program-options-dev \
        wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root

RUN wget http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/13005/l_mkl_2018.3.222.tgz && \
    tar xf l_mkl_2018.3.222.tgz && \
    rm l_mkl_2018.3.222.tgz && \
    cd l_mkl_2018.3.222 && \
    sed -i 's/ACCEPT_EULA=decline/ACCEPT_EULA=accept/g' silent.cfg && \
    sed -i 's/ARCH_SELECTED=ALL/ARCH_SELECTED=INTEL64/g' silent.cfg && \
    ./install.sh -s silent.cfg

RUN wget --no-check-certificate http://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz && \
    tar xf 3.3.4.tar.gz && \
    rm 3.3.4.tar.gz && \
    mv eigen-eigen-5a0156e40feb eigen

ARG CXX_FLAGS
ENV CXX_FLAGS=${CXX_FLAGS:--mavx2}

COPY . ctranslate-dev
WORKDIR /root/ctranslate-dev
RUN mkdir build && \
    cd build && \
    cmake -DEIGEN_ROOT=/root/eigen -DCMAKE_INSTALL_PREFIX=/root/ctranslate \
          -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
          -DCMAKE_C_COMPILER=gcc-8 -DCMAKE_CXX_COMPILER=g++-8 .. && \
    VERBOSE=1 make -j4 && \
    make install

WORKDIR /root
RUN cp -r /opt/intel/mkl/lib/intel64/* /root/ctranslate/lib && \
    rm /root/ctranslate/lib/*.a && \
    rm /root/ctranslate/lib/*intel* /root/ctranslate/lib/*sequential* /root/ctranslate/lib/*tbb*

FROM ubuntu:16.04

COPY --from=builder /root/ctranslate /root/ctranslate

RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        libstdc++6 \
        libboost-program-options1.58.0 && \
    apt-get autoremove -y software-properties-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/ctranslate/lib
