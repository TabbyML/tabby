#!/bin/bash

SOURCE_FILE=$2
OUTPUT_FILE=$3

onmt-main --model_dir /averaged-ende-ckpt500k-v2 \
          --model_type Transformer \
          --config /config.yml \
          --gpu_allow_growth \
          infer \
          --features_file $SOURCE_FILE --predictions_file $OUTPUT_FILE
