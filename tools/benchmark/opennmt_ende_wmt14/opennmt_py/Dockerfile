FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV VERSION=3.0.4
RUN pip install --no-cache-dir OpenNMT-py==$VERSION

ADD convertv2_v3.py .
RUN wget -q https://opennmt-models.s3.amazonaws.com/transformer-ende-wmt-pyOnmt.tar.gz && \
    tar xf *.tar.gz && \
    rm *.tar.gz && \
    python3 convertv2_v3.py -v2model averaged-10-epoch.pt -v3model averaged-10-epoch-v3.pt && \
    rm averaged-10-epoch.pt

COPY tokenize detokenize translate /
