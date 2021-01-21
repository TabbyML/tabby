#!/bin/bash

MODEL_DIR="/averaged-ende-ckpt500k-v2"
SOURCE_FILE=$1
OUTPUT_FILE=$2
BATCH_SIZE=$3
BEAM_SIZE=$4

echo "model_dir: $MODEL_DIR" >> config.yml
echo "data:" >> config.yml
echo "  source_vocabulary: $MODEL_DIR/wmtende.vocab" >> config.yml
echo "  target_vocabulary: $MODEL_DIR/wmtende.vocab" >> config.yml
echo "params:" >> config.yml
echo "  beam_width: $BEAM_SIZE" >> config.yml
echo "infer:" >> config.yml
echo "  batch_size: $BATCH_SIZE" >> config.yml

onmt-main --model_type Transformer --config config.yml --auto_config --gpu_allow_growth \
          infer \
          --log_prediction_time --features_file $SOURCE_FILE --predictions_file $OUTPUT_FILE \
          2>&1 | grep "Tokens per second" | awk '{print $NF}'

rm config.yml
