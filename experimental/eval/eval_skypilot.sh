#!/bin/bash
set -ex

ARGS="tabby-eval skypilot.yaml --env MAX_RECORDS=1000"

if ! sky exec $ARGS; then
  sky launch -c $ARGS
fi

scp tabby-eval:~/sky_workdir/reports.ipynb ./
scp tabby-eval:~/sky_workdir/reports.html ./
