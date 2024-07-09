---
title: Enhanced Tabby Deployment on Modal
authors:
  - name: moqimoqidea
    url: https://github.com/moqimoqidea
    image_url: https://github.com/moqimoqidea
tags: [deployment]
---

# Enhanced Tabby Deployment on Modal: Utilizing Persistent Volumes and Model Caching

Today we’re diving deeper into our latest deployment updates on Modal, focusing on two critical enhancements: model caching and the use of persistent volumes. These features are designed to optimize both the scalability and usability of Tabby in serverless environments.

## Understanding Model Caching

One of the significant updates in our Modal deployment strategy is the implementation of a model cache directory. This change is crucial for a few reasons:

1. **Scalability and Speed:** The most substantial parts of our deployment are the model files, which are often large. By caching these files in the image layer, we ensure that the container does not need to re-download the model every time it starts. This dramatically reduces the startup and shutdown times, making our service highly responsive and cost-effective—ideal for Function as a Service (FaaS) scenarios. More on image caching can be found in Modal's guide on [Image caching and rebuilds](https://modal.com/docs/guide/custom-container#image-caching-and-rebuilds).

2. **Efficiency:** With model caching, the overall efficiency of the deployment improves because the time and resources spent on fetching and loading models are minimized. This setup is particularly beneficial in environments where rapid scaling is necessary.

### Use Model Caching

Here's how we use Modal's image cache to accelerate the deployment and scaling of services, When the app is first run, Modal will download these models that we need to use and caches the final image.

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

app = App("tabby-server", image=image)
```

## The Role of Persistent Volumes

Persistent volumes (PVs) are another cornerstone of our updated deployment strategy. Their use addresses several operational challenges:

1. **Data Persistence:** In a typical FaaS setup, where containers are frequently started and stopped, maintaining user data and custom configurations across sessions is challenging. Persistent volumes solve this by ensuring that data such as user-generated content, configurations, and indices remain intact across container restarts. For more details, see Modal's section on [persisting volumes](https://modal.com/docs/guide/volumes#persisting-volumes).

2. **User Experience:** By synchronizing configuration files and other essential data, PVs enhance the user experience. They eliminate the need to reconfigure settings or regenerate data, thus providing a seamless service experience. This is especially valuable for users with custom configurations who expect consistent performance and reliability.

3. **Operational Stability:** PVs provide a stable storage solution that copes well with the high-frequency start-stop nature of serverless environments. This stability is crucial for maintaining service reliability and performance.

### Use of Persistent Volumes

Here's how we use Modal's persistent volumes to keep data synchronized in a FaaS environment and independent of the container's lifecycle. The `create_if_missing=True` parameter ensures that the volume is created lazily, only if it doesn’t exist already. The `_allow_background_volume_commits` parameter ensures that every few seconds the attached Volume will be snapshotted and its new changes committed. A final snapshot and commit is also automatically performed on container shutdown.

```python
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
```

## The Complete App.py

The `app.py` script is at the heart of our Modal deployment. It integrates all configurations, model management, and service functionalities into a cohesive application. Here is it:

```python
"""Usage:
modal serve app.py

To force a rebuild by pulling the latest image tag, use:
MODAL_FORCE_BUILD=1 modal serve app.py
"""

import os

from modal import Image, App, asgi_app, gpu, Volume

IMAGE_NAME = "tabbyml/tabby"
MODEL_ID = "TabbyML/StarCoder-1B"
CHAT_MODEL_ID = "TabbyML/Qwen2-1.5B-Instruct"
EMBEDDING_MODEL_ID = "TabbyML/Nomic-Embed-Text"
GPU_CONFIG = gpu.L4()

TABBY_BIN = "/opt/tabby/bin/tabby"
TABBY_ENV = os.environ.copy()
TABBY_ENV['TABBY_MODEL_CACHE_ROOT'] = '/models'


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
            "1",
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

For the most up-to-date version of `app.py`, please visit the [Tabby GitHub repository: Modal app.py](https://github.com/TabbyML/tabby/blob/main/website/docs/quick-start/installation/modal/app.py).

## Conclusion

These enhancements to our deployment strategy on Modal not only improve the operational aspects of running Tabby but also significantly enhance the user experience by providing faster startup times and data persistence. By integrating model caching and persistent volumes, we ensure that Tabby remains a robust and efficient solution in the dynamic landscape of serverless computing.

For those looking to implement similar strategies, we encourage exploring the detailed configurations and benefits discussed in our [full tutorial](https://github.com/TabbyML/tabby/blob/main/website/docs/quick-start/installation/modal/index.md), which provides a step-by-step guide on setting up your Tabby instance with these advanced features.

We hope this post provides valuable insights into our deployment improvements and inspires you to optimize your applications similarly. Stay tuned for more updates and happy coding with Tabby!
