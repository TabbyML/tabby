#!/bin/bash
set -ex

cd clients

openapi-python-client generate \
  --path ../website/static/openapi.json \
  --config ../experimental/openapi/python.yaml \
  --meta setup
