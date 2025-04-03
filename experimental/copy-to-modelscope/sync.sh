#!/bin/bash

set -ex


MODELS=$(cat <<EOF
TabbyML/StarCoder-1B
TabbyML/StarCoder-3B
TabbyML/StarCoder-7B
TabbyML/CodeLlama-7B
TabbyML/CodeLlama-13B
TabbyML/WizardCoder-3B
EOF
)

for i in $MODELS; do
  ./main.sh $i $1
done

