#!/bin/bash

set -ex

rm -rf tabby-python-client

curl http://localhost:8080/api-docs/openapi.json | jq 'delpaths([
  ["paths", "/v1beta/chat/completions"]
])' > /tmp/openapi.json

openapi-python-client generate \
  --path /tmp/openapi.json \
  --config ./python.yaml \
  --meta setup
