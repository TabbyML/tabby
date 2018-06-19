FROM ubuntu:16.04 as builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        g++-8 \
        gcc-8 \
        libboost-program-options-dev \
        libboost-python-dev \
        python-pip \
        wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root

RUN wget https://github.com/intel/mkl-dnn/releases/download/v0.14/mklml_lnx_2018.0.3.20180406.tgz && \
    tar xf mklml_lnx_2018.0.3.20180406.tgz && \
    rm mklml_lnx_2018.0.3.20180406.tgz && \
    mv mklml_lnx_2018.0.3.20180406 mklml

ARG CXX_FLAGS
ENV CXX_FLAGS=${CXX_FLAGS}

COPY . ctranslate-dev
WORKDIR /root/ctranslate-dev
RUN mkdir build && \
    cd build && \
    cmake -DMKLML_ROOT=/root/mklml -DCMAKE_INSTALL_PREFIX=/root/ctranslate \
          -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
          -DCMAKE_C_COMPILER=gcc-8 -DCMAKE_CXX_COMPILER=g++-8 .. && \
    VERBOSE=1 make -j4 && \
    make install

WORKDIR /root/ctranslate-dev/python
RUN pip --no-cache-dir install setuptools wheel
RUN CTRANSLATE_ROOT=/root/ctranslate python setup.py bdist_wheel

WORKDIR /root
RUN cp -r /root/mklml/lib/libiomp5.so /root/ctranslate/lib && \
    cp -r /root/mklml/lib/libmklml_intel.so /root/ctranslate/lib && \
    cp /root/ctranslate-dev/python/dist/*whl /root/ctranslate

FROM ubuntu:16.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libstdc++6 \
        libboost-program-options1.58.0 \
        libboost-python1.58.0 \
        python-pip && \
    apt-get autoremove -y software-properties-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/ctranslate /root/ctranslate

RUN pip --no-cache-dir install setuptools
RUN pip --no-cache-dir install /root/ctranslate/ctranslate-0.1.0-cp27-cp27mu-linux_x86_64.whl

WORKDIR /root

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/ctranslate/lib
