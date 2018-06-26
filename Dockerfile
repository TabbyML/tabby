FROM ubuntu:16.04 as builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        cpio \
        libboost-program-options-dev \
        libboost-python-dev \
        python-pip \
        wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root

RUN wget http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/13005/l_mkl_2018.3.222.tgz
RUN tar xf l_mkl_2018.3.222.tgz && \
    rm l_mkl_2018.3.222.tgz && \
    cd l_mkl_2018.3.222 && \
    sed -i 's/ACCEPT_EULA=decline/ACCEPT_EULA=accept/g' silent.cfg && \
    sed -i 's/ARCH_SELECTED=ALL/ARCH_SELECTED=INTEL64/g' silent.cfg && \
    ./install.sh -s silent.cfg

COPY . ctranslate-dev

ARG CXX_FLAGS
ENV CXX_FLAGS=${CXX_FLAGS}

WORKDIR /root/ctranslate-dev
RUN mkdir build && \
    cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/root/ctranslate \
          -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" .. && \
    VERBOSE=1 make -j4 && \
    make install

WORKDIR /root/ctranslate-dev/python
RUN pip --no-cache-dir install setuptools wheel
RUN CTRANSLATE_ROOT=/root/ctranslate python setup.py bdist_wheel

WORKDIR /root
RUN cp /opt/intel/lib/intel64/libiomp5.so /root/ctranslate/lib && \
    cp /root/ctranslate-dev/python/dist/*whl /root/ctranslate

FROM ubuntu:16.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libboost-program-options1.58.0 \
        libboost-python1.58.0 \
        python-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/ctranslate /root/ctranslate

RUN pip --no-cache-dir install setuptools
RUN pip --no-cache-dir install /root/ctranslate/ctranslate-0.1.0-cp27-cp27mu-linux_x86_64.whl

WORKDIR /root

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/ctranslate/lib
