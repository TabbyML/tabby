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
* [`tasks`](./tasks): Various data processing scripts.
* [`tabformer`](./tabformer): *NOT RELEASED* Trainer(PEFT w/RLHF) for tabby models.

## Development

Assuming Linux workstation with:
1. docker
2. docker w/ gpu driver
3. python 3.10

Use `make setup-development-environment` to setup basic dev environment, and `make dev` to start local development server.


## 🙋 We're hiring
Come help us make Tabby even better. We're growing fast [and would love for you to join us](https://tabbyml.notion.site/Careers-35b1a77f3d1743d9bae06b7d6d5b814a).
