#!/bin/bash

SOURCE_FILE=$2
OUTPUT_FILE=$3

onmt-main --model_type Transformer --config /config.yml --auto_config --gpu_allow_growth \
          infer \
          --features_file $SOURCE_FILE --predictions_file $OUTPUT_FILE
