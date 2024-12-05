from __future__ import annotations

from asgi_proxy import asgi_proxy

import os
import time
import bentoml
import socket
import subprocess


class TabbyServer:
    def __init__(self, model_id: str, chat_model_id: str) -> None:
        self.launcher = subprocess.Popen(
            [
                "tabby",
                "serve",
                "--model",
                model_id,
                "--chat-model",
                chat_model_id,
                "--device",
                "cuda",
                "--port",
                "8000",
            ]
        )

    def ready(self) -> bool:
        try:
            socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
            return True
        except (socket.timeout, ConnectionRefusedError):
            # Check if launcher webserving process has exited.
            # If so, a connection can never be made.
            retcode = self.launcher.poll()
            if retcode is not None:
                raise RuntimeError(f"launcher exited unexpectedly with code {retcode}")
            return False

    def wait_until_ready(self) -> None:
        while not self.ready():
            time.sleep(1.0)


app = asgi_proxy("http://127.0.0.1:8000")


@bentoml.service(
    resources={"gpu": 1, "gpu_type": "nvidia-l4"},
    traffic={"timeout": 10},
)
@bentoml.mount_asgi_app(app, path="/")
class Tabby:
    @bentoml.on_deployment
    def prepare():
        download_tabby_dir("tabby-local")

    @bentoml.on_shutdown
    def shutdown(self):
        upload_tabby_dir("tabby-local")

    def __init__(self) -> None:
        model_id = "StarCoder-1B"
        chat_model_id = "Qwen2-1.5B-Instruct"

        # Start the server subprocess.
        self.server = TabbyServer(model_id, chat_model_id)

        # Wait for the server to be ready.
        self.server.wait_until_ready()


def download_tabby_dir(username: str) -> None:
    """Download the tabby directory for the given user."""
    # Ensure the bucket `tabby-cloud-managed` and the path `users/tabby-local` exist in your R2 storage
    if os.system(f"rclone sync r2:/tabby-cloud-managed/users/{username} ~/.tabby") == 0:
        print("Tabby directory downloaded successfully.")
    else:
        raise RuntimeError("Failed to download tabby directory")


def upload_tabby_dir(username: str) -> None:
    """Upload the tabby directory for the given user."""
    if os.system(f"rclone sync --links ~/.tabby r2:/tabby-cloud-managed/users/{username}") == 0:
        print("Tabby directory uploaded successfully.")
    else:
        raise RuntimeError("Failed to upload tabby directory")
