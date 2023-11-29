"""Usage:
modal serve app.py
"""

import os
from modal import Image, Stub, asgi_app

GPU_CONFIG = os.environ.get("GPU_CONFIG", "T4")
IMAGE_NAME = "tabbyml/tabby:0.6.0"
MODEL_ID = os.environ.get("MODEL_ID", "TabbyML/StarCoder-1B")
PARALLELISM = os.environ.get("PARALLELISM", "4")


def download_model():
    import os
    import subprocess

    model_id = os.environ.get("MODEL_ID")
    subprocess.run(
        [
            "/opt/tabby/bin/tabby",
            "download",
            "--model",
            model_id,
        ]
    )


image = (
    Image.from_registry(
        IMAGE_NAME,
        add_python="3.11",
    )
    .env({"MODEL_ID": MODEL_ID})
    .dockerfile_commands("ENTRYPOINT []")
    .run_function(download_model)
    .pip_install("asgi-proxy-lib")
    .env({"PARALLELISM": PARALLELISM})
)

stub = Stub("tabby-server-loadtest", image=image)


@stub.function(
    gpu=GPU_CONFIG,
    allow_concurrent_inputs=int(PARALLELISM),
    container_idle_timeout=120,
    timeout=360,
)
@asgi_app()
def app():
    import os
    import socket
    import subprocess
    import time
    from asgi_proxy import asgi_proxy

    model_id = os.environ.get("MODEL_ID")
    parallelism = os.environ.get("PARALLELISM")

    env = os.environ.copy()
    env["TABBY_DISABLE_USAGE_COLLECTION"] = "1"

    launcher = subprocess.Popen(
        [
            "/opt/tabby/bin/tabby",
            "serve",
            "--model",
            model_id,
            "--port",
            "8000",
            "--device",
            "cuda",
            "--parallelism",
            parallelism,
        ],
        env=env
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
