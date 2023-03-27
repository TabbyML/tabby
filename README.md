<div align="center">
# 🐾 Tabby

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Docker build status](https://img.shields.io/github/actions/workflow/status/TabbyML/tabby/docker.yml?label=docker%20image%20build)
</div>

> **Warning**
> Tabby is still in the alpha phrase

An opensource / on-prem alternative to GitHub Copilot.

## Features

* Self-contained, with no need for a DBMS or cloud service
* Web UI for visualizing and configuration models and MLOps.
* OpenAPI interface, easy to integrate with existing infrastructure (e.g Cloud IDE).
* Consumer level GPU supports (FP-16 weight loading with various optimization).

## Getting started
To get started with Tabby, see the [docs/deployment](./docs/deployment).

## Roadmap

* [ ] Fine-tuning models on private code repository.
* [ ] Plot metrics in admin panel (e.g acceptance rate).
* [ ] Production ready (Open Telemetry, Prometheus metrics).
