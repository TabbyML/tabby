from pathlib import Path

import modal
from modal import Image, Mount, Secret, Stub, asgi_app, gpu, method

GPU_CONFIG = gpu.T4()
MODEL_ID = "TabbyML/StarCoder-1B"
LAUNCH_FLAGS = ["serve", "--model", MODEL_ID, "--port", "8000", "--device", "cuda"]


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
        "tabbyml/tabby@sha256:64d71ec4c7d9ae7269e6301ad4106baad70ee997408691a6af17d7186283a856",
        add_python="3.11",
    )
    .dockerfile_commands("ENTRYPOINT []")
    .run_function(download_model)
    .pip_install(
        "git+https://github.com/TabbyML/tabby.git#egg=tabby-python-client&subdirectory=experimental/eval/tabby-python-client"
    )
)

stub = Stub("tabby-" + MODEL_ID.split("/")[-1], image=image)


@stub.cls(
    gpu=GPU_CONFIG,
    allow_concurrent_inputs=10,
    container_idle_timeout=60 * 10,
    timeout=360,
)
class Model:
    def __enter__(self):
        import socket
        import subprocess
        import time

        from tabby_python_client import Client

        self.launcher = subprocess.Popen(["/opt/tabby/bin/tabby"] + LAUNCH_FLAGS)
        self.client = Client("http://127.0.0.1:8000")

        # Poll until webserver at 127.0.0.1:8000 accepts connections before running inputs.
        def webserver_ready():
            try:
                socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
                return True
            except (socket.timeout, ConnectionRefusedError):
                # Check if launcher webserving process has exited.
                # If so, a connection can never be made.
                retcode = self.launcher.poll()
                if retcode is not None:
                    raise RuntimeError(
                        f"launcher exited unexpectedly with code {retcode}"
                    )
                return False

        while not webserver_ready():
            time.sleep(1.0)

        print("Tabby server ready!")

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.launcher.terminate()

    @method()
    async def health(self):
        from tabby_python_client.api.v1 import health

        resp = await health.asyncio(client=self.client)
        return resp.to_dict()

    @method()
    async def complete(self, language: str, prompt: str):
        from tabby_python_client.api.v1 import completion
        from tabby_python_client.models import (
            CompletionRequest,
            DebugOptions,
            CompletionResponse,
            Segments,
        )

        request = CompletionRequest(
            language=language, debug_options=DebugOptions(raw_prompt=prompt)
        )
        resp: CompletionResponse = await completion.asyncio(
            client=self.client, json_body=request
        )
        return resp.choices[0].text


@stub.local_entrypoint()
def main():
    import json

    model = Model()
    print(model.health.remote())

    with open("./output.jsonl", "w") as fout:
        with open("./sample.jsonl") as fin:
            for line in fin:
                x = json.loads(line)
                prompt = x["crossfile_context"]["text"] + x["prompt"]
                label = x["groundtruth"]
                prediction = model.complete.remote("python", prompt)

                json.dump(dict(prompt=prompt, label=label, prediction=prediction), fout)
