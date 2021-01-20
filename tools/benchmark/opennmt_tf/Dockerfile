FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-distutils \
        wget \
        && \
    wget -nv https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q https://opennmt-models.s3.amazonaws.com/averaged-ende-ckpt500k-v2.tar.gz && \
    tar xf *.tar.gz && \
    rm *.tar.gz

RUN python3 -m pip install --no-cache-dir tensorflow==2.4.* OpenNMT-tf==2.14.*

COPY opennmt_tf/run.sh /

ENTRYPOINT ["/run.sh"]
