FROM nvcr.io/nvidia/pytorch:22.09-py3

RUN git clone --branch v5.3 https://github.com/NVIDIA/FasterTransformer.git && \
    cd FasterTransformer && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON .. && \
    make -j$(nproc)

RUN wget -q https://opennmt-models.s3.amazonaws.com/transformer-ende-wmt-pyOnmt.tar.gz && \
    tar xf *.tar.gz && \
    rm *.tar.gz

RUN pip install --no-cache-dir OpenNMT-py==1.1.1

WORKDIR FasterTransformer/build

# Patch to fix translation of empty files.
RUN sed -i '121i\ \ \ \ \ \ \ \ ex_fields = {}' /opt/conda/lib/python3.8/site-packages/onmt/inputters/dataset_base.py
RUN sed -i 's/report_time=True/report_time=False/g' ../examples/pytorch/decoding/translate_example.py

COPY tokenize detokenize translate /
