FROM tensorflow/tensorflow:2.11.0-gpu

RUN curl -O https://opennmt-models.s3.amazonaws.com/averaged-ende-ckpt500k-v2.tar.gz && \
    tar xf *.tar.gz && \
    rm *.tar.gz
RUN cd averaged-ende-ckpt500k-v2 && \
    curl -O https://opennmt-trainingdata.s3.amazonaws.com/wmtende.model

RUN python3 -m pip install --no-cache-dir OpenNMT-tf==2.31.*

COPY *.yml /
COPY tokenize detokenize translate /
