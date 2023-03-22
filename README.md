## 🐾 Tabby
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Docker build status](https://img.shields.io/github/actions/workflow/status/TabbyML/tabby/docker.yml?label=docker%20image%20build)

> **Warning**
> This repository is undering heavy construction, everything changes fast.

An opensource / on-prem alternative to GitHub Copilot

## Contents
* [`admin`](./admin): Admin panel for monitoring / settings purpose.
* [`server`](./server): API server for completion requests. It also logs users' selections (as feedback to model's quality).
* [`deployment`](./deployment): Container related deployment configs.
* [`converter`](./converter): Converts a [transformers](https://huggingface.co/docs/transformers) causal LM model into TensorRT / FasterTransformer serving formats.
* [`preprocess`](./preprocess): Preprocess files into [datasets](https://huggingface.co/docs/datasets)
* [`tabformer`](./tabformer): *NOT RELEASED* Trainer(PEFT w/RLHF) for tabby models.
