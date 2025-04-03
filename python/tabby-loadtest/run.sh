#!/bin/bash

record() {
  echo $GPU_CONFIG,$MODEL_ID,$PARALLELISM,$1 >> record.csv
}

cleanup() {
MODAL_APP_ID=$(modal app list | grep tabby-server-loadtest | grep deployed | awk '{print $2}')

if [ -z $MODAL_APP_ID ]; then
  modal app stop $MODAL_APP_ID
fi
}

loadtest() {
export GPU_CONFIG=$1
export MODEL_ID=$2
export PARALLELISM=$3

>&2 modal deploy server.py

export MODAL_PROCESS_ID=$!
export TABBY_API_HOST=https://wsxiaoys--tabby-server-loadtest-app.modal.run

# wait for warmup
>&2 echo "Waiting for warmup..."


n=0
while [[ "$(curl -s -o /dev/null -w ''%{http_code}'' $TABBY_API_HOST/v1/health)" != "200" ]]; do
  if [ "$n" -ge 5 ]; then
    # error after 5 retries.
    return 1
  fi

  sleep 10;
  n=$((n+1)) 
done

>&2 echo "Start load testing..."

>&2 k6 run loadtest.js
SUCCESS=$?
METRICS=$(cat metrics.txt)
rm metrics.txt

if [ $SUCCESS -ne 0 ]; then
  record $METRICS,FAILED
else
  record $METRICS,SUCCESS
fi

cleanup

return $SUCCESS
}

function dichotomic_search {
  min=$1
  max=$2
  command=$3

  while (( $min < $max )); do
    # Compute the mean between min and max, rounded up to the superior unit
    current=$(( (min + max + 1 ) / 2 ))
    
    if $command $current
      then min=$current
      else max=$((current - 1))
    fi
  done
}

test_t4() {
  loadtest T4 $MODEL_ID $1
}

test_a10g() {
  loadtest A10G $MODEL_ID $1
}

test_a100() {
  loadtest A100 $MODEL_ID $1
}

test_1b3b_model() {
  export MODEL_ID="$1"

  dichotomic_search 1 12 test_t4
  dichotomic_search 1 32 test_a10g
  dichotomic_search 1 64 test_a100
}

test_7b_model() {
  export MODEL_ID="$1"

  dichotomic_search 1 8 test_a100
}

test_13b_model() {
  export MODEL_ID="$1"

  dichotomic_search 1 8 test_a100
}

# test_7b_model TabbyML/CodeLlama-7B
test_13b_model TabbyML/CodeLlama-13B