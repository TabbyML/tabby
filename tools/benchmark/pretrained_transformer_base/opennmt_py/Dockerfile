FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q https://opennmt-models.s3.amazonaws.com/transformer-ende-wmt-pyOnmt.tar.gz && \
    tar xf *.tar.gz && \
    rm *.tar.gz

# TODO: remove --no-deps once OpenNMT-py includes PyTorch 1.9 in its dependencies.
RUN pip install --no-cache-dir --no-deps OpenNMT-py==2.1.2 && \
    pip install --no-cache-dir torchtext==0.5.* configargparse==1.* pyonmttok==1.*

COPY *.sh /
