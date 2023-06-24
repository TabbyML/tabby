#!/bin/bash
set -x

docker-compose up -d

while ! curl -X POST http://localhost:8080/v1/health; do
  echo "server not ready, waiting..."
  sleep 5
fi

echo done
