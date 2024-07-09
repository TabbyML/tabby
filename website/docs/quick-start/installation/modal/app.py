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
