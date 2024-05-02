import os
from modal import Image, Stub, gpu, asgi_app, Volume

IMAGE_NAME = os.environ.get("TABBY_IMAGE", "tabbyml/tabby")

image = (
    Image.from_registry(
        IMAGE_NAME,
        add_python="3.11",
    )
    .dockerfile_commands("ENTRYPOINT []")
    .pip_install("asgi-proxy-lib")
)

stub = Stub("tabby-demo-server", image=image)
volume = Volume.from_name("tabby-demo-server-volume", create_if_missing=True)

@stub.function(
    concurrency_limit=1,
    allow_concurrent_inputs=100,
    container_idle_timeout=600*2,
    timeout=600,
    volumes = {"/data": volume},
    _allow_background_volume_commits=True
)
@asgi_app()
def entry():
    import json
    import socket
    import subprocess
    import time
    import os
    from asgi_proxy import asgi_proxy

    env = os.environ.copy()
    env["TABBY_DISABLE_USAGE_COLLECTION"] = "1"
    env["TABBY_WEBSERVER_DEMO_MODE"] = "1"

    chat_model = dict(
        kind="openai-chat",
        model_name="deepseek-coder",
        api_endpoint="https://api.deepseek.com/v1",
        api_key=env.get("OPENAI_API_KEY", ""),
    )

    launcher = subprocess.Popen(
        [
            "/opt/tabby/bin/tabby-cpu",
            "serve",
            "--port",
            "8000",
            "--chat-device",
            "experimental-http"
            "--chat-model",
            json.dumps(chat_model),
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