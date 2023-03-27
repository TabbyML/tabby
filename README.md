<div align="center">

# 🐾 Tabby

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Docker build status](https://img.shields.io/github/actions/workflow/status/TabbyML/tabby/docker.yml?label=docker%20image%20build)

![architecture](https://user-images.githubusercontent.com/388154/227848286-f5b697f2-9df7-4adc-9a2a-84b029788030.png)

</div>

> **Warning**
> Tabby is still in the alpha phrase

An opensource / on-prem alternative to GitHub Copilot.

## Features

* Self-contained, with no need for a DBMS or cloud service
* Web UI for visualizing and configuration models and MLOps.
* OpenAPI interface, easy to integrate with existing infrastructure (e.g Cloud IDE).
* Consumer level GPU supports (FP-16 weight loading with various optimization).

## Get started

### Docker

The easiest way of getting started is using the `deployment/docker-compose.yml`:
```bash
docker-compose up
```
Note: To use GPUs, you need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). We also recommend using NVIDIA drivers with CUDA version 11.8 or higher.

You can then query the server using `/v1/completions` endpoint:
```bash
curl -X POST http://localhost:5000/v1/completions -H 'Content-Type: application/json' --data '{
    "prompt": "def binarySearch(arr, left, right, x):\n    mid = (left +"
}'
```

We also provides an interactive playground in admin panel [localhost:8501](http://localhost:8501)

![image](https://user-images.githubusercontent.com/388154/227792390-ec19e9b9-ebbb-4a94-99ca-8a142ffb5e46.png)

### API documentation

Tabby opens an FastAPI server at [localhost:5000](https://localhost:5000), which embeds an OpenAPI documentation of the HTTP API.

## Development

Go to `development` directory.
```
make dev
```
or
```
make triton-dev
```

## TODOs

* [ ] Fine-tuning models on private code repository.
* [ ] Plot metrics in admin panel (e.g acceptance rate).
* [ ] Production ready (Open Telemetry, Prometheus metrics).
* [ ] Token streaming using Server-Sent Events (SSE)
