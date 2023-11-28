import asyncio
import json
import os
import pandas as pd

from datetime import datetime
from modal import Image, Stub, gpu, method
from typing import List, Optional, Tuple


GPU_CONFIG = gpu.A10G()

MODEL_ID = os.environ.get("MODEL_ID")
LAUNCH_FLAGS = ["serve", "--model", MODEL_ID, "--port", "8000", "--device", "cuda"]


def download_model():
    import subprocess
    import os
    MODEL_ID = os.environ.get("MODEL_ID")
    print(f'MODEL_ID={MODEL_ID}')
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
        "tabbyml/tabby:0.5.5",
        add_python="3.11",
    )
    .env({"MODEL_ID": os.environ.get("MODEL_ID")})
    .dockerfile_commands("ENTRYPOINT []")
    .copy_local_dir(local_path='./modal/tabby_python_client/tabby_python_client', remote_path='/root/tabby_python_client')
    .pip_install(
        "httpx",
        "pandas"
    )
    .run_function(download_model)
)

stub = Stub("tabby-" + MODEL_ID.split("/")[-1], image=image)


@stub.cls(
    gpu=GPU_CONFIG,
    concurrency_limit=10,
    allow_concurrent_inputs=2,
    container_idle_timeout=60 * 10,
    timeout=600,
)
class Model:

    def __enter__(self):
        import socket
        import subprocess
        import os
        import time

        from tabby_python_client import Client
        

        my_env = os.environ.copy()
        my_env["TABBY_DISABLE_USAGE_COLLECTION"] = "1"
        MODEL_ID = os.environ.get("MODEL_ID")
        print(f'MODEL_ID={MODEL_ID}')
       
        LAUNCH_FLAGS = ["serve", "--model", MODEL_ID, "--port", "8000", "--device", "cuda"]
        self.launcher = subprocess.Popen(["/opt/tabby/bin/tabby"] + LAUNCH_FLAGS, env=my_env)
        self.client = Client("http://127.0.0.1:8000", timeout=240)

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
    async def complete(self, language: str, index: int, prompt: str) -> Tuple[int, Optional[str], Optional[str]]:
        from tabby_python_client import errors
        from tabby_python_client.api.v1 import completion
        from tabby_python_client.models import (
            CompletionRequest,
            DebugOptions,
        )
        from tabby_python_client.types import Response


        request = CompletionRequest(
            language=language, debug_options=DebugOptions(raw_prompt=prompt)
        )

        try:
            resp: Response = await completion.asyncio_detailed(
                client=self.client, json_body=request
            )
        
            if resp.parsed is not None:
                return index, resp.parsed.choices[0].text, None
            else:
                return index, None, f"<{resp.status_code}>"
        except errors.UnexpectedStatus as e:
            return index, None, f"error: code={e.status_code} content={e.content} error={e}"
        except Exception as e:
            return index, None, f"error type: {type(e)}"

def write_log(log: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('./modal/log.txt', 'a') as f:
        f.write(f"{now} : {log}")
        f.write("\n")

def chunker(seq, size) -> List:
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def read_dataframe_from_file(language: str, file: str) -> pd.DataFrame:
    whole_path_file = "./data/" + MODEL_ID.split("/")[-1] + "/" + language + "/" + file
    objs = []
    with open(whole_path_file) as fin:
        for line in fin:
            obj = json.loads(line)
            if 'crossfile_context' in obj.keys():
                obj['raw_prompt'] = obj['crossfile_context']['text'] + obj['prompt']
            else:
                obj['raw_prompt'] = obj['prompt']
            objs.append(obj)

    df = pd.DataFrame(objs)
    return df

@stub.local_entrypoint()
async def main(language: str, files: str):
    #Multiple files seperated by ','
    
    model = Model()

    health_resp = model.health.remote()
    print(f'model info:\n{health_resp}')
    assert(health_resp['model'] == MODEL_ID)

    files = files.split(',')

    for file in files:
  
        df = read_dataframe_from_file(language, file.strip())
        
        write_log(f'model: {MODEL_ID}; language: {language}; file: {file}: length = {len(df)}')

        
        if 'prediction' in df.columns:
            df_no_prediction = df[df['prediction'].isna()]
        else:
            df_no_prediction = df

        skipped = len(df) - len(df_no_prediction)
        success = 0
        error = 0

        for group in chunker(df_no_prediction, 30):
            outputs = await asyncio.gather(*[model.complete.remote.aio(language, index, row['raw_prompt']) for index, row in group.iterrows()])

            for index, prediction, error_msg in outputs:
                if prediction is not None:
                    df.loc[index, 'prediction'] = prediction
                    success += 1
                else:
                    df.loc[index, 'error'] = error_msg
                    error += 1

        write_log(f"Skipped {skipped} rows, {success} rows with predictions, {error} rows with errors")

        whole_path_file = "./data/" + MODEL_ID.split("/")[-1] + "/" + language + "/" + file
        with open(whole_path_file, 'w') as fout:
            for index, row in df.iterrows():
                row_dict = row.to_dict()
                json.dump(row_dict, fout)
                fout.write('\n')

        write_log(f"model: {MODEL_ID}; language: {language}; file: {file}: end!\n")
