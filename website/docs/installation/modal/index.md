# Modal

[Modal](https://modal.com/) is a serverless GPU provider. By leveraging Modal, your Tabby instance will run on demand. When there are no requests to the Tabby server for a certain amount of time, Modal will schedule the container to sleep, thereby saving GPU costs.

## Setup

First we import the components we need from `modal`.

```python
from modal import Image, Stub, asgi_app, gpu
```

Next, we set the base docker image version, which model to serve, taking care to specify the GPU configuration required to fit the model into VRAM.

```python
IMAGE_NAME = "tabbyml/tabby"
MODEL_ID = "TabbyML/StarCoder-1B"
GPU_CONFIG = gpu.T4()
```

Currently supported GPUs in Modal:

- `T4`: Low-cost GPU option, providing 16GiB of GPU memory.
- `L4`: Mid-tier GPU option, providing 24GiB of GPU memory.
- `A100`: The most powerful GPU available in the cloud. Available in 40GiB and 80GiB GPU memory configurations.
- `A10G`: A10G GPUs deliver up to 3.3x better ML training performance, 3x better ML inference performance, and 3x better graphics performance, in comparison to NVIDIA T4 GPUs.
- `Any`: Selects any one of the GPU classes available within Modal, according to availability.

For detailed usage, please check official [Modal GPU reference](https://modal.com/docs/reference/modal.gpu).

## Define the container image

We want to create a Modal image which has the Tabby model cache pre-populated. The benefit of this is that the container no longer has to re-download the model - instead, it will take advantage of Modal’s internal filesystem for faster cold starts.

### Download the weights

```python
def download_model():
    import subprocess

    subprocess.run(
        [
            "/opt/tabby/bin/tabby",
            "download",
            "--model",
            MODEL_ID,
        ]
    )
```


### Image definition

We’ll start from an image by tabby, and override the default ENTRYPOINT for Modal to run its own which enables seamless serverless deployments.

Next we run the download step to pre-populate the image with our model weights.

Finally, we install the `asgi-proxy-lib` to interface with modal's asgi webserver over localhost.

```python
image = (
    Image.from_registry(
        IMAGE_NAME,
        add_python="3.11",
    )
    .dockerfile_commands("ENTRYPOINT []")
    .run_function(download_model)
    .pip_install("asgi-proxy-lib")
)
```

### The app function

The endpoint function is represented with Modal's `@stub.function`. Here, we:

1. Launch the Tabby process and wait for it to be ready to accept requests.
2. Create an ASGI proxy to tunnel requests from the Modal web endpoint to the local Tabby server.
3. Specify that each container is allowed to handle up to 10 requests simultaneously.
4. Keep idle containers for 2 minutes before spinning them down.

```python
stub = Stub("tabby-server-" + MODEL_ID.split("/")[-1], image=image)
@stub.function(
    gpu=GPU_CONFIG,
    allow_concurrent_inputs=10,
    container_idle_timeout=120,
    timeout=360,
)
@asgi_app()
def app():
    import socket
    import subprocess
    import time
    from asgi_proxy import asgi_proxy

    launcher = subprocess.Popen(
        [
            "/opt/tabby/bin/tabby",
            "serve",
            "--model",
            MODEL_ID,
            "--port",
            "8000",
            "--device",
            "cuda",
        ]
    )

    # Poll until webserver at 127.0.0.1:8000 accepts connections before running inputs.
    def tabby_ready():
        try:
            socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
            return True
        except (socket.timeout, ConnectionRefusedError):
            # Check if launcher webserving process has exited.
            # If so, a connection can never be made.
            retcode = launcher.poll()
            if retcode is not None:
                raise RuntimeError(f"launcher exited unexpectedly with code {retcode}")
            return False

    while not tabby_ready():
        time.sleep(1.0)

    print("Tabby server ready!")
    return asgi_proxy("http://localhost:8000")
```

### Serve the app

Once we deploy this model with `modal serve app.py`, it will output the url of the web endpoint, in a form of `https://<USERNAME>--tabby-server-starcoder-1b-app-dev.modal.run`.

To test if the server is working, you can send a post request to the web endpoint.

```shell
curl --location 'https://<USERNAME>--tabby-server-starcoder-1b-app-dev.modal.run/v1/completions' \
--header 'Content-Type: application/json' \
--data '{
  "language": "python",
  "segments": {
    "prefix": "def fib(n):\n    ",
    "suffix": "\n        return fib(n - 1) + fib(n - 2)"
  }
}'
```

If you can get json response like in the following case, the app server is up and have fun!

```json
{
    "id": "cmpl-4196b0c7-f417-4c48-9329-4a56aa86baea",
    "choices": [
        {
            "index": 0,
            "text": "if n == 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:"
        }
    ]
}
```



![App Running](./app-running.png)

Now it can be used as tabby server url in tabby editor extensions!
See [app.py](https://github.com/TabbyML/tabby/blob/main/website/docs/installation/modal/app.py) for the full code used in this tutorial. 
