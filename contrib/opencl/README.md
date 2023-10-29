# Tabby with OpenCL


## Build the docker container

```sh
cd contrib/opencl
docker build -t tabbyml/tabby:opencl -f Dockerfile ../../
```

## Running the Container

```sh
docker run -it \
  -p 8080:8080 \
  -v $HOME/.tabby:/data \
  --gpus all \
  --device /dev/dri \
  tabbyml/tabby:opencl \
  serve \
  --model TabbyML/StarCoder-1B
```

## Test the Tabby API endpoint

```sh
curl -L 'http://127.0.0.1:8080/v1/completions' \
-H 'Content-Type: application/json' \
-H 'Accept: application/json' \
-d '{
  "language": "python",
  "segments": {
    "prefix": "def fib(n):\n    ",
    "suffix": "\n        return fib(n - 1) + fib(n - 2)"
  }
}'
```