from pathlib import Path

import modal
from modal import Image, Mount, Secret, Stub, asgi_app, gpu, method
import os
import pandas as pd

import asyncio

GPU_CONFIG = gpu.A100()
MODEL_ID =  os.environ.get("MODEL_ID", "TabbyML/StarCoder-3B")
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
        "tabbyml/tabby:0.5.0",
        add_python="3.11",
    )
    .dockerfile_commands("ENTRYPOINT []")
    .run_function(download_model)
    .pip_install(
        "git+https://github.com/TabbyML/tabby.git#egg=tabby-python-client&subdirectory=experimental/eval/tabby-python-client",
        "pandas"
    )
)

stub = Stub("tabby-" + MODEL_ID.split("/")[-1], image=image)


@stub.cls(
    gpu=GPU_CONFIG,
    concurrency_limit=10,
    allow_concurrent_inputs=1,
    container_idle_timeout=60 * 10,
    timeout=360,
)
class Model:
    def __enter__(self):
        import socket
        import subprocess, os
        import time

        from tabby_python_client import Client

        my_env = os.environ.copy()
        my_env["TABBY_DISABLE_USAGE_COLLECTION"] = "1"
        self.launcher = subprocess.Popen(["/opt/tabby/bin/tabby"] + LAUNCH_FLAGS, env=my_env)
        self.client = Client("http://127.0.0.1:8000", timeout=60)

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
    async def complete(self, language, crossfile_context, index, row):
        from tabby_python_client.api.v1 import completion
        from tabby_python_client.models import (
            CompletionRequest,
            DebugOptions,
            CompletionResponse,
            Segments,
        )
        from tabby_python_client.types import Response
        from tabby_python_client import errors

        if 'prediction' in row and not pd.isnull(row['prediction']):
            return None, None, None

        if crossfile_context:
            prompt = row["crossfile_context"]["text"] + row["prompt"]
        else:
            prompt = row["prompt"]

        groundtruth = row["groundtruth"]
       
        request = CompletionRequest(
            language=language, debug_options=DebugOptions(raw_prompt=prompt)
        )
        # resp: CompletionResponse = await completion.asyncio(
        #     client=self.client, json_body=request
        # )
        try:
            resp: Response = await completion.asyncio_detailed(
                client=self.client, json_body=request
            )
        
            if resp.parsed != None:
                return index, resp.parsed.choices[0].text, None
            else:
                return index, None, f"<{resp.status_code}>"
        except errors.UnexpectedStatus as e:
            return index, None, f"error: code={e.status_code} content={e.content} error={e}"
        except Exception as e:
            return index, None, f"error type: {type(e)}"



@stub.local_entrypoint()
async def main(language, file):
    import json

    print(MODEL_ID)

    model = Model()
    print(model.health.remote())

    whole_path_file = "./data/" + MODEL_ID.split("/")[-1] + "/" + language + "/" + file

    if file == 'line_completion.jsonl':
        crossfile_context = False
    else:
        crossfile_context = True

    objs = []
    with open(whole_path_file) as fin:
        for line in fin:
            obj = json.loads(line)
            objs.append(obj)

    df = pd.DataFrame(objs)

    outputs = await asyncio.gather(*[model.complete.remote.aio(language, crossfile_context, index, row) for index, row in df.iterrows()])

    skipped = 0
    success = 0
    error = 0

    for index, prediction, error_msg in outputs:
        if index is None:
            skipped += 1
        elif prediction is not None:
            df.loc[index, 'prediction'] = prediction
            success += 1
        else:
            df.loc[index, 'error'] = error_msg
            error += 1
    print(f"Skipped {skipped} rows, {success} rows with predictions, {error} rows with errors")

    with open(whole_path_file, 'w') as fout:
        for index, row in df.iterrows():
            json.dump(row.to_dict(), fout)
            fout.write('\n')
                