# Docker

There is a supplied docker image to make deploying a server as a container easier.


## CPU

**Command line**
```bash
docker run \
  -p 8080:8080 -v $HOME/.tabby:/data \
  tabbyml/tabby serve --model TabbyML/SantaCoder-1B
```


**Docker Compose**
```yaml
version: '3.5'

services:
  tabby:
    restart: always
    image: tabbyml/tabby
    command: serve --model TabbyML/SantaCoder-1B
    volumes:
      - "$HOME/.tabby:/data"
    ports:
      - 8080:8080
```

## CUDA (requires NVIDIA Container Toolkit)

**Command line**
```bash
docker run \
  --gpus all -p 8080:8080 -v $HOME/.tabby:/data \
  tabbyml/tabby \
  serve --model TabbyML/SantaCoder-1B --device cuda
```

**Docker Compose**
```yaml
version: '3.5'
services:
  tabby:
    restart: always
    image: tabbyml/tabby
    command: serve --model TabbyML/SantaCoder-1B --device cuda
    volumes:
      - "$HOME/.tabby:/data"
    ports:
      - 8080:8080
    resources:
    reservations:
      devices:
      - driver: nvidia
        count: 1
        capabilities: [gpu]
```
