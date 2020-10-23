#!/bin/bash

onmt_translate -model /workspace/averaged-10-epoch.pt -report_time $@ \
               2>&1 | grep "Tokens per second" | awk '{print $NF}'
