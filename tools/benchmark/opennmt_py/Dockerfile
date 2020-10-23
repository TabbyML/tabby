FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q https://opennmt-models.s3.amazonaws.com/transformer-ende-wmt-pyOnmt.tar.gz && \
    tar xf *.tar.gz && \
    rm *.tar.gz sentencepiece.model

RUN pip install OpenNMT-py==2.0.0rc1

COPY opennmt_py/run.sh /

ENTRYPOINT ["/run.sh"]
