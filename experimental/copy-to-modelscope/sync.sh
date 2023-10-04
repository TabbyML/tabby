#!/bin/bash

set -ex

# TabbyML/StarCoder-3B

MODELS=$(cat <<EOF
TabbyML/StarCoder-7B
TabbyML/WizardCoder-1B
TabbyML/WizardCoder-3B
TabbyML/CodeLlama-7B
TabbyML/CodeLlama-13B
TabbyML/StarCoder-1B
EOF
)

for i in $MODELS; do
  ./main.sh $i $1
done

