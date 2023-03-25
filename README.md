## 🐾 Tabby
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Docker build status](https://img.shields.io/github/actions/workflow/status/TabbyML/tabby/docker.yml?label=docker%20image%20build)

> **Warning**
> This repository is undering heavy construction, everything changes fast.

An opensource / on-prem alternative to GitHub Copilot

## Development

Assuming Linux workstation with:
1. docker
2. docker w/ gpu driver
3. python 3.10

Use `make setup-development-environment` to setup basic dev environment, and `make dev` to start local development server.

## Deployment
1. `make up-triton`
2. Open Admin Panel [http://localhost:8501](http://localhost:8501)
3. Types some code in editor!

![Screenshot](https://user-images.githubusercontent.com/388154/227720116-938cc6e9-e6cd-435b-a4d2-7b846a2101c7.png)

## 🙋 We're hiring
Come help us make Tabby even better. We're growing fast [and would love for you to join us](https://tabbyml.notion.site/Careers-35b1a77f3d1743d9bae06b7d6d5b814a).
