FROM opennmt/ctranslate2:1.17.0-ubuntu18-cuda11.0 as model_converter

RUN wget -q https://opennmt-models.s3.amazonaws.com/transformer-ende-wmt-pyOnmt.tar.gz && \
    tar xf *.tar.gz && \
    rm *.tar.gz

RUN pip install --no-cache-dir OpenNMT-py==2.0.0

RUN ct2-opennmt-py-converter --model_path averaged-10-epoch.pt --model_spec TransformerBase --output_dir /model
RUN wget -q -P /model https://opennmt-models.s3.amazonaws.com/vmap.txt
RUN cp sentencepiece.model /model

FROM opennmt/ctranslate2:1.17.0-ubuntu18-cuda11.0

COPY --from=model_converter /model /model

RUN pip install --no-cache-dir pyonmttok==1.*

COPY *.sh /
