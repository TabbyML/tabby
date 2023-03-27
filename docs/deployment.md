# Deployment

## Prerequisites

You need install following dependencies
* docker `>= 17.06`
* An NVIDIA GPU with enough VRAM to run the model you want.
* [NVIDIA Docker Driver](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)

## Installation

### Tabby Server

1. Goto [`deployment`](../deployment) directory
2. Execute `docker-compose up`.

### Tabby Client

There're several ways to talk to the Tabby Server.

#### Tabby Admin Panel

Open Admin Panel [http://localhost:8501](http://localhost:8501)

![image](https://user-images.githubusercontent.com/388154/227792390-ec19e9b9-ebbb-4a94-99ca-8a142ffb5e46.png)

#### OpenAPI

Open [http://localhost:5000](http://localhost:5000) to view the OpenAPI documents.
