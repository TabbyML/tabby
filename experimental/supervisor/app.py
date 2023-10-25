"""Usage:
    python app.py --model TabbyML/StarCoder-1B --chat-model TabbyML/Mistral-7B --device metal
"""

import socket
import time
import asyncio
import argparse
import uvicorn
import sys
import subprocess
from asgi_proxy import asgi_proxy

MODEL_ID = "TabbyML/StarCoder-1B"


class TabbyLauncher(object):
    def __init__(self, args):
        self.proc = None
        self.args = args

    def start(self):
        print("Starting tabby process...")
        self.proc = subprocess.Popen(
            [
                "tabby",
                "serve",
            ]
            + self.args
            + [
                "--port",
                "8081",
            ],
        )

        while not self._server_ready():
            time.sleep(1.0)
        return self

    def _server_ready(self):
        # Poll until webserver at 127.0.0.1:8081 accepts connections before running inputs.
        try:
            socket.create_connection(("127.0.0.1", 8081), timeout=1).close()
            print("Tabby server ready!")
            return True
        except (socket.timeout, ConnectionRefusedError):
            # Check if launcher webserving process has exited.
            # If so, a connection can never be made.
            retcode = self.proc.poll()
            if retcode is not None:
                raise RuntimeError(f"launcher exited unexpectedly with code {retcode}")
            return False

    @property
    def is_running(self):
        return self.proc is not None

    def stop(self):
        if self.proc is None:
            return

        self.proc.terminate()
        self.proc = None
        print("Tabby process stopped.")


class Timer:
    def __init__(self, timeout, callback):
        self._timeout = timeout
        self._callback = callback
        self._task = asyncio.ensure_future(self._job())

    async def _job(self):
        await asyncio.sleep(self._timeout)
        self._callback()

    def cancel(self):
        self._task.cancel()


def supervisor(serve_args):
    launcher = TabbyLauncher(serve_args)
    proxy = asgi_proxy("http://localhost:8081")
    timer = None

    async def callback(scope, receive, send):
        nonlocal timer

        if not launcher.is_running:
            launcher.start()
        elif timer is not None:
            timer = timer.cancel()

        timer = Timer(600, launcher.stop)
        return await proxy(scope, receive, send)

    return callback


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a tabby supervisor")
    parser.add_argument(
        "-p", "--port", type=int, default=8080, help="Port to use (default: 8080)"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )

    args, serve_args = parser.parse_known_args()

    app = supervisor(serve_args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
