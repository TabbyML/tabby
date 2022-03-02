#!/bin/bash
onmt-detokenize-text --tokenizer_config /tokenization.yml < $1 > $2
