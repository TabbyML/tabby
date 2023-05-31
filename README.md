<div align="center">

# 🐾 Tabby

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Docker build status](https://img.shields.io/github/actions/workflow/status/TabbyML/tabby/docker.yml?label=docker%20image%20build)](https://github.com/TabbyML/tabby/actions/workflows/docker.yml)
[![Docker pulls](https://img.shields.io/docker/pulls/tabbyml/tabby)](https://hub.docker.com/r/tabbyml/tabby)

</div>

Self-hosted AI coding assistant. An opensource / on-prem alternative to GitHub Copilot.

> **Warning**
> Tabby is still in the alpha phase

## Features

* Self-contained, with no need for a DBMS or cloud service
* Web UI for visualizing and configuration models and MLOps.
* OpenAPI interface, easy to integrate with existing infrastructure (e.g Cloud IDE).
* Consumer level GPU supports (FP-16 weight loading with various optimization).

## Demo
<p align="center">
  <a href="https://huggingface.co/spaces/TabbyML/tabby"><img alt="Open in Spaces" src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-md.svg"></a>
</p>

<p align="center">
  <img alt="Demo" src="https://user-images.githubusercontent.com/388154/230440226-9bc01d05-9f57-478b-b04d-81184eba14ca.gif">
</p>



## Get started: Server

### Docker

**NOTE**: Tabby requires [Pascal or newer](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) NVIDIA GPU.

Before running Tabby, ensure the installation of the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
We suggest using NVIDIA drivers that are compatible with CUDA version 11.8 or higher.
```bash
# Create data dir and grant owner to 1000 (Tabby run as uid 1000 in container)
mkdir -p tabby_data && chown -R 1000 tabby_data

# Serve the model.
docker run --gpus all -it --rm \
  -v "/$(pwd)/data:/data" \
  -p 8080:8080 \
  -e TABBY_ROOT=/data \
  --name=tabby \
  tabbyml/tabby serve \
  --model=TabbyML/J-350M
```

You can then query the server using `/v1/completions` endpoint:
```bash
curl -X POST http://localhost:8080/v1/completions -H 'Content-Type: application/json' --data '{
    "prompt": "def binarySearch(arr, left, right, x):\n    mid = (left +"
}'
```

### Skypilot
See [deployment/skypilot/README.md](./deployment/skypilot/README.md)

## Getting Started: Client
We offer multiple methods to connect to Tabby Server, including using OpenAPI and editor extensions.

### API
Tabby has opened a FastAPI server at [localhost:8080](https://localhost:8080), which includes an OpenAPI documentation of the HTTP API. The same API documentation is also hosted at https://tabbyml.github.io/tabby

### Editor Extensions

* [VSCode Extension](./clients/vscode) – Install from the [marketplace](https://marketplace.visualstudio.com/items?itemName=TabbyML.vscode-tabby), or [open-vsx.org](https://open-vsx.org/extension/TabbyML/vscode-tabby)
* [VIM Extension](./clients/vim)

## Development

Go to `development` directory.
```bash
make dev
```
or
```bash
make dev-triton # Turn on triton backend (for cuda env developers)
```
