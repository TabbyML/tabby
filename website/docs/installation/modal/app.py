"""Usage:
modal serve app.py
"""

from modal import Image, Stub, asgi_app, gpu

IMAGE_NAME = "tabbyml/tabby"
MODEL_ID = "TabbyML/StarCoder-1B"
GPU_CONFIG = gpu.T4()


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


image = (
    Image.from_registry(
        IMAGE_NAME,
        add_python="3.11",
    )
    .dockerfile_commands("ENTRYPOINT []")
    .run_function(download_model)
    .pip_install("asgi-proxy-lib")
)

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
            "--parallelism",
            "4",
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
