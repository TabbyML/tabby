#!/bin/bash
set -ex

ARGS="tabby-eval skypilot.yaml --env MAX_RECORDS=100"

if ! sky exec $ARGS; then
  sky launch -c $ARGS
fi

scp tabby-eval:~/sky_workdir/reports.html ./
