FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q https://opennmt-models.s3.amazonaws.com/transformer-ende-wmt-pyOnmt.tar.gz && \
    tar xf *.tar.gz && \
    rm *.tar.gz sentencepiece.model

# TODO: remove --no-deps once OpenNMT-py includes PyTorch 1.7 in its dependencies.
RUN pip install --no-cache-dir --no-deps OpenNMT-py==2.0.0 && \
    pip install --no-cache-dir torchtext==0.5.* configargparse==1.*

COPY opennmt_py/run.sh /

ENTRYPOINT ["/run.sh"]
