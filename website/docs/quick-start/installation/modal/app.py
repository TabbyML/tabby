"""Usage:
modal serve app.py

To force a rebuild by pulling the latest image tag, use:
MODAL_FORCE_BUILD=1 modal serve app.py
"""

from modal import Image, App, asgi_app, gpu

IMAGE_NAME = "tabbyml/tabby"
MODEL_ID = "TabbyML/StarCoder-1B"
CHAT_MODEL_ID = "TabbyML/Qwen2-1.5B-Instruct"
EMBEDDING_MODEL_ID = "TabbyML/Nomic-Embed-Text"
GPU_CONFIG = gpu.T4()

TABBY_BIN = "/opt/tabby/bin/tabby"


def download_model(model_id: str):
    import subprocess

    subprocess.run(
        [
            TABBY_BIN,
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
    .dockerfile_commands("ENTRYPOINT []")
    .run_function(download_model, kwargs={"model_id": EMBEDDING_MODEL_ID})
    .run_function(download_model, kwargs={"model_id": CHAT_MODEL_ID})
    .run_function(download_model, kwargs={"model_id": MODEL_ID})
    .pip_install("asgi-proxy-lib")
)

app = App("tabby-server", image=image)


@app.function(
    gpu=GPU_CONFIG,
    allow_concurrent_inputs=10,
    container_idle_timeout=120,
    timeout=360,
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
