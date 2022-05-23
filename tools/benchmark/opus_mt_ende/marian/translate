#!/bin/bash

DEVICE=$1
SOURCE_FILE=$2
OUTPUT_FILE=$3
GEMM=${GEMM_TYPE:-float32}
MODEL=/model/opus.spm32k-spm32k.transformer-align.model1.npz.best-perplexity.npz

EXTRA_ARGS=""
if [ $DEVICE = "CPU" ]; then
    EXTRA_ARGS+=" --cpu-threads $OMP_NUM_THREADS"
fi

if [ $GEMM = "float16" ]; then
    EXTRA_ARGS+=" --fp16"
elif [ $GEMM != "float32" ]; then
    /root/marian-dev/build/marian-conv -f $MODEL -t /tmp/model.bin -g $GEMM
    MODEL=/tmp/model.bin
fi

/root/marian-dev/build/marian-decoder \
    -m $MODEL \
    -v /model/opus.spm32k-spm32k.vocab.yml /model/opus.spm32k-spm32k.vocab.yml \
    -b 4 --mini-batch 32 --maxi-batch 512 --maxi-batch-sort src -w 2500 \
    --quiet --quiet-translation \
    $EXTRA_ARGS < $SOURCE_FILE > $OUTPUT_FILE
