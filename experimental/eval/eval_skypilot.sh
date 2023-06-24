#!/bin/bash
set -ex

if ! sky exec tabby-eval skypilot.yaml; then
  sky launch -c tabby-eval skypilot.yaml
fi

scp tabby-eval:~/sky_workdir/reports.html ./
