FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

ENV MODEL_NAME="Helsinki-NLP/opus-mt-en-de"

RUN python3 -m pip install --no-cache-dir transformers==4.26.* sentencepiece sacremoses

COPY tokenize detokenize translate /

# Call translate once to download and cache the model.
RUN touch /tmp/input.txt && /translate CPU /tmp/input.txt /tmp/output.txt && rm /tmp/input.txt
