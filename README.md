<div align="center">

# 🐾 Tabby

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Docker build status](https://img.shields.io/github/actions/workflow/status/TabbyML/tabby/docker.yml?label=docker%20image%20build)
[![Huggingface Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-spaces-blue)](https://huggingface.co/spaces/TabbyML/tabby)

![architecture](https://user-images.githubusercontent.com/388154/229353706-230d70e1-7d09-48e2-a884-4da768bccf6f.png)

</div>

Self-hosted AI coding assistant. An opensource / on-prem alternative to GitHub Copilot.

> **Warning**
> Tabby is still in the alpha phrase

## Features

* Self-contained, with no need for a DBMS or cloud service
* Web UI for visualizing and configuration models and MLOps.
* OpenAPI interface, easy to integrate with existing infrastructure (e.g Cloud IDE).
* Consumer level GPU supports (FP-16 weight loading with various optimization).

## Get started

### Docker

The easiest way of getting started is using the official docker image:
```bash
# Create data dir and grant owner to 1000 (Tabby run as uid 1000 in container)
mkdir -p data/hf_cache && chown -R 1000 data

docker run \
  -it --rm \
  -v ./data:/data \
  -v ./data/hf_cache:/home/app/.cache/huggingface \
  -p 5000:5000 \
  -e MODEL_NAME=TabbyML/J-350M \
  tabbyml/tabby
```

To use the GPU backend (triton) for a faster inference speed:
```bash
docker run \
  --gpus all \
  -it --rm \
  -v ./data:/data \
  -v ./data/hf_cache:/home/app/.cache/huggingface \
  -p 5000:5000 \
  -e MODEL_NAME=TabbyML/J-350M \
  -e MODEL_BACKEND=triton \
  tabbyml/tabby
```
Note: To use GPUs, you need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). We also recommend using NVIDIA drivers with CUDA version 11.8 or higher.

You can then query the server using `/v1/completions` endpoint:
```bash
curl -X POST http://localhost:5000/v1/completions -H 'Content-Type: application/json' --data '{
    "prompt": "def binarySearch(arr, left, right, x):\n    mid = (left +"
}'
```

We also provides an interactive playground in admin panel [localhost:5000/_admin](http://localhost:5000/_admin)

![image](https://user-images.githubusercontent.com/388154/227792390-ec19e9b9-ebbb-4a94-99ca-8a142ffb5e46.png)

### Skypilot
See [deployment/skypilot/README.md](./deployment/skypilot/README.md)

## API documentation

Tabby opens an FastAPI server at [localhost:5000](https://localhost:5000), which embeds an OpenAPI documentation of the HTTP API.

## Development

Go to `development` directory.
```bash
make dev
```
or
```bash
make dev-triton # Turn on triton backend (for cuda env developers)
```

## TODOs

* [ ] VIM Client [#36](https://github.com/TabbyML/tabby/issues/36)
* [ ] Fine-tuning models on private code repository. [#23](https://github.com/TabbyML/tabby/issues/23)
* [ ] Production ready (Open Telemetry, Prometheus metrics).
