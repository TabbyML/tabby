#!/bin/bash

SOURCE_FILE=$2
OUTPUT_FILE=$3

/opt/ctranslate2/bin/translate --model /model --src $SOURCE_FILE --out $OUTPUT_FILE --device auto --batch_size 32 --beam_size 4 --compute_type ${COMPUTE_TYPE:-default}
