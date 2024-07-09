# Modal

[Modal](https://modal.com/) is a serverless GPU provider. By leveraging Modal, your Tabby instance will run on demand. When there are no requests to the Tabby server for a certain amount of time, Modal will schedule the container to sleep, thereby saving GPU costs.

## Setup

First we import the components we need from `modal`.

```python
import os
from modal import Image, App, asgi_app, gpu, Volume
```

Next, we set the base Docker image version and specify which model to serve. Initially, we considered using the `T4` GPU configuration for its cost-effectiveness. However, during testing, we encountered a significant number of HTTP 204 responses, indicating potential limitations with the T4's capabilities for our needs. As a result, we are now using the `L4` GPU configuration to efficiently utilize VRAM, which offers a better balance between performance and cost for our application.

```python
IMAGE_NAME = "tabbyml/tabby"
MODEL_ID = "TabbyML/StarCoder-1B"
CHAT_MODEL_ID = "TabbyML/Qwen2-1.5B-Instruct"
EMBEDDING_MODEL_ID = "TabbyML/Nomic-Embed-Text"
GPU_CONFIG = gpu.L4()

TABBY_BIN = "/opt/tabby/bin/tabby"
TABBY_ENV = os.environ.copy()
TABBY_ENV['TABBY_MODEL_CACHE_ROOT'] = '/models'
```

Currently supported GPUs in Modal:

- `T4`: Low-cost GPU option, providing 16GiB of GPU memory.
- `L4`: Mid-tier GPU option, providing 24GiB of GPU memory.
- `A100`: The most powerful GPU available in the cloud. Available in 40GiB and 80GiB GPU memory configurations.
- `H100`: The flagship data center GPU of the Hopper architecture. Enhanced support for FP8 precision and a Transformer Engine that provides up to 4X faster training over the prior generation for GPT-3 (175B) models.
- `A10G`: A10G GPUs deliver up to 3.3x better ML training performance, 3x better ML inference performance, and 3x better graphics performance, in comparison to NVIDIA T4 GPUs.
- `Any`: Selects any one of the GPU classes available within Modal, according to availability.

For detailed usage, please check official [Modal GPU reference](https://modal.com/docs/reference/modal.gpu).

## Define the container image

We want to create a Modal image which has the Tabby model cache pre-populated.
The benefit of this is that the container no longer has to re-download the model—instead,
it will take advantage of Modal’s internal filesystem for faster cold starts.

### Download the weights

```python
def download_model():
    import subprocess

    subprocess.run(
        [
            TABBY_BIN,
            "download",
            "--model",
            MODEL_ID,
        ],
        env=TABBY_ENV,
    )


def download_chat_model():
    import subprocess

    subprocess.run(
        [
            TABBY_BIN,
            "download",
            "--model",
            CHAT_MODEL_ID,
        ],
        env=TABBY_ENV,
    )


def download_embedding_model():
    import subprocess

    subprocess.run(
        [
            TABBY_BIN,
            "download",
            "--model",
            EMBEDDING_MODEL_ID,
        ],
        env=TABBY_ENV,
    )
```

### Image definition

We’ll start from an image by tabby, and override the default ENTRYPOINT for Modal to run its own which enables seamless serverless deployments.

Next, we run the download step to pre-populate the image with our model weights.

Finally, we install the `asgi-proxy-lib` to interface with Modal's ASGI webserver over localhost.

```python
image = (
    Image.from_registry(
        IMAGE_NAME,
        add_python="3.11",
    )
    .dockerfile_commands("ENTRYPOINT []")
    .run_function(download_model)
    .run_function(download_chat_model)
    .run_function(download_embedding_model)
    .pip_install("asgi-proxy-lib")
)
```

### The app function

The endpoint function is represented with Modal's `@app.function`. Here, we:

1. Launch the Tabby process and wait for it to be ready to accept requests.
2. Create an ASGI proxy to tunnel requests from the Modal web endpoint to the local Tabby server.
3. Specify that each container is allowed to handle up to 10 requests simultaneously.
4. Keep idle containers for 2 minutes before spinning them down.

```python
app = App("tabby-server", image=image)

data_volume = Volume.from_name("tabby-data", create_if_missing=True)
data_dir = "/data"

@app.function(
    gpu=GPU_CONFIG,
    allow_concurrent_inputs=10,
    container_idle_timeout=120,
    timeout=360,
    volumes={data_dir: data_volume},
    _allow_background_volume_commits=True,
    concurrency_limit=1,
)
@asgi_app()
def app_serve():
    import socket
    import subprocess
    import time
    from asgi_proxy import asgi_proxy

    launcher = subprocess.Popen(
        [
            TABBY_BIN,
            "serve",
            "--model",
            MODEL_ID,
            "--chat-model",
            CHAT_MODEL_ID,
            "--port",
            "8000",
            "--device",
            "cuda",
            "--parallelism",
            "4",
        ],
        env=TABBY_ENV,
    )

    # Poll until webserver at 127.0.0.1:8000 accepts connections before running inputs.
    def tabby_ready():
        try:
            socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
            return True
        except (socket.timeout, ConnectionRefusedError):
            # Check if a launcher webservice process has exited.
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

Once we deploy this model with `modal serve app.py`, it will output the URL of the web endpoint, in the form of `https://<USERNAME>--tabby-server-app-serve-dev.modal.run`.

If you encounter any issues, particularly related to caching, you can force a rebuild by running `MODAL_FORCE_BUILD=1 modal serve app.py`. This ensures that the latest image tag is used by ignoring cached layers.

![App Running](./app-running.png)

Now it can be used as a tabby server URL in Tabby editor extensions!
See [app.py](https://github.com/TabbyML/tabby/blob/main/website/docs/quick-start/installation/modal/app.py) for the full code used in this tutorial.
