## Prerequisites

You need install following dependencies
* docker `>= 17.06`
* An NVIDIA GPU with enough VRAM to run the model you want.
* [NVIDIA Docker Driver](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)

## Setup Tabby Server with `docker-compose`.

1. Goto [`deployment`](../deployment) directory
2. Execute `docker-compose up`.

## Tabby Client

There're several ways to talk to the Tabby Server.

### Tabby Admin Panel [http://localhost:8501](http://localhost:8501)

![image](https://user-images.githubusercontent.com/388154/227792390-ec19e9b9-ebbb-4a94-99ca-8a142ffb5e46.png)

### OpenAPI [http://localhost:5000](http://localhost:5000)

![image](https://user-images.githubusercontent.com/388154/227835790-29e21eb5-6e9c-45ab-aa0f-c4c7ce399ad7.png)
