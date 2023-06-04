FROM opennmt/ctranslate2:3.6.0-ubuntu20.04-cuda11.2 as model_converter

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q https://opennmt-models.s3.amazonaws.com/transformer-ende-wmt-pyOnmt.tar.gz && \
    tar xf *.tar.gz && \
    rm *.tar.gz

RUN pip install --no-cache-dir OpenNMT-py==2.2.0

RUN ct2-opennmt-py-converter --model_path averaged-10-epoch.pt --output_dir /model
RUN wget -q -P /model https://opennmt-models.s3.amazonaws.com/vmap.txt
RUN cp sentencepiece.model /model

FROM opennmt/ctranslate2:3.6.0-ubuntu20.04-cuda11.2

COPY --from=model_converter /model /model

RUN pip install --no-cache-dir pyonmttok==1.*

COPY tokenize detokenize translate /
