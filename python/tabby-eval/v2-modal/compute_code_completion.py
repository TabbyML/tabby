import argparse
import json
import logging
import os
import signal
import subprocess
import threading
import time

import httpx
import pandas as pd
from pandas import json_normalize

from compute_metrics import evaluation

EMBEDDING_MODEL_ID = "TabbyML/Nomic-Embed-Text"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def check_service_health(endpoint, token):
    def modal_tabby_ready():
        url = "{}/v1/health".format(endpoint)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f'Bearer {token}'
        }

        try:
            response = httpx.get(url=url, headers=headers, timeout=5)
            if response.status_code == 200:
                logging.info("Server details: {}".format(response.json()))
                return True
            else:
                return False
        except Exception as e:
            logging.error(f"Failed to check service health: {e}")
            return False

    while not modal_tabby_ready():
        time.sleep(5)

    logging.info("Modal tabby server ready!")


def monitor_serve_output(process):
    while True:
        line = process.stdout.readline()
        if not line:
            break
        logging.info(line.strip())


def start_tabby_server(model):
    logging.info("Starting tabby server for model {model}".format(model=model))

    modal_env = os.environ.copy()
    modal_env["MODEL_ID"] = model
    modal_env["EMBEDDING_MODEL_ID"] = EMBEDDING_MODEL_ID

    process = subprocess.Popen(args=["modal", "serve", "app.py"],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               text=True,
                               env=modal_env)

    threading.Thread(target=monitor_serve_output, args=(process,)).start()

    return process


def send_sigint_to_process(process):
    try:
        os.kill(process.pid, signal.SIGINT)
        logging.info("SIGINT signal sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send SIGINT signal: {e}")


def send_request_with_retry(url, headers, payload, timeout=10, max_retries=10):
    retries = 0
    response = None

    while retries < max_retries:
        try:
            response = httpx.post(url=url, headers=headers, content=json.dumps(payload), timeout=timeout)
            if response.status_code == 200:
                return response
            else:
                retries += 1
                time.sleep(1)
        except httpx.RequestError as e:
            logging.error(f"Get code completion failed: {e}")
            retries += 1
            time.sleep(1)

    return response


def parse_model_id(model_id: str):
    parts = model_id.split('/')
    if len(parts) == 1:
        return "TabbyML", parts[0]
    elif len(parts) == 2:
        return parts[0], parts[1]
    else:
        raise ValueError(f"Invalid model id {model_id}")


def get_model_prompt_template(model_id: str):
    registry, model_name = parse_model_id(model_id)
    url = f"https://raw.githubusercontent.com/{registry}/registry-tabby/main/models.json"

    response = httpx.get(url=url)
    response.raise_for_status()

    for model in response.json():
        if model["name"] == model_name:
            return model.get("prompt_template", None)

    return None


def generate_predictions(endpoint: str,
                         token: str,
                         jsonl_file: str,
                         prediction_jsonl_file: str,
                         data_function,
                         prompt_template: str = None):
    logging.info(f"Generating predictions to {prediction_jsonl_file}...")
    df = pd.read_json(jsonl_file, lines=True)
    df_flat = json_normalize(df.to_dict(orient="records"))

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {token}'
    }

    predictions = []
    prediction_status = []
    for index, row in df_flat.iterrows():
        data = data_function(row, prompt_template) if prompt_template else data_function(row)

        # TODO: Add parallelism support
        url = f"{endpoint}/v1/completions"
        response = send_request_with_retry(url, headers, data, timeout=10, max_retries=10)

        if response.status_code == 200:
            predictions.append(response.json()['choices'][0]['text'])
            prediction_status.append("success")
        else:
            predictions.append("Request failed after retry.")
            prediction_status.append("failed")

    df_flat['prediction'] = predictions
    df_flat['prediction_status'] = prediction_status
    df_success = df_flat[df_flat['prediction_status'] == "success"]

    total_records = len(df_flat)
    success_count = len(df_success)
    failed_count = total_records - success_count
    logging.info(f"Total predictions: {total_records}, Success: {success_count}, Failed: {failed_count}")

    df_success.to_json(prediction_jsonl_file, orient='records', lines=True)


def default_data_function(row):
    return {
        "language": row['language'],
        "segments": {
            "prefix": row['segments.prefix'],
            "suffix": row['segments.suffix']
        }
    }


def cross_file_data_function(row, prompt_template):
    crossfile_context_list = row['segments.crossfile_context.list']
    sorted_list = sorted(crossfile_context_list, key=lambda x: x['score'])

    # TODO: The added prefix comments should be consistent with the language, for the time being use "//" for all
    combined_context = "\n".join(
        f"// Path: {item['filename']}\n" + "\n".join(f"// {line}" for line in item['retrieved_chunk'].split("\n"))
        for item in sorted_list
    ) + "\n"
    logging.debug(f"Combined context in cross_file_data_function: {combined_context}")

    return {
        "debug_options": {
            "raw_prompt": prompt_template.format(
                prefix=combined_context + row['segments.prefix'],
                suffix=row['segments.suffix'])
        }
    }


def eval_code_completion(endpoint: str,
                         token: str,
                         model: str,
                         jsonl_file: str,
                         prediction_jsonl_file: str,
                         evaluation_jsonl_file: str,
                         cross_file_content_prediction_jsonl_file: str,
                         cross_file_content_evaluation_jsonl_file: str,
                         start_tabby_server_on_modal: bool):
    # Start modal tabby server
    process = None
    if start_tabby_server_on_modal:
        process = start_tabby_server(model)

    # Check the service health
    logging.info("Checking service health...")
    check_service_health(endpoint, token)

    # Generate predictions
    logging.info("Generating predictions...")
    generate_predictions(endpoint, token, jsonl_file, prediction_jsonl_file, default_data_function)
    generate_predictions(endpoint, token, jsonl_file,
                         cross_file_content_prediction_jsonl_file,
                         cross_file_data_function,
                         get_model_prompt_template(model))
    logging.info("Predictions generated!")

    # Run the evaluation
    logging.info("Running evaluation...")
    evaluation(prediction_jsonl_file, evaluation_jsonl_file)
    evaluation(cross_file_content_prediction_jsonl_file, cross_file_content_evaluation_jsonl_file)
    logging.info("Evaluation completed")

    # Stop the server
    if start_tabby_server_on_modal and process:
        logging.info("Stopping server...")
        send_sigint_to_process(process)
        logging.info("Server stopped!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval tabby code completion.")
    parser.add_argument("--endpoint", type=str, required=True, help="tabby server endpoint.")
    parser.add_argument("--token", type=str, required=True, default="", help="tabby server token.")
    parser.add_argument("--model", type=str, required=True, help="evaluation model.")
    parser.add_argument("--jsonl_file", type=str, default="data.jsonl", help="evaluation jsonl file.")
    parser.add_argument("--output_jsonl_file_prefix", type=str, help="""
    output jsonl file prefix, it will generate four files: 
    prediction, evaluation, cross_file_content_prediction, cross_file_content_evaluation.
    """)
    parser.add_argument("--start_tabby_server_on_modal", type=str, default="1",
                        help="start tabby server on modal manager, accepts 1 or another.")

    args = parser.parse_args()
    output_jsonl_file_prefix = args.output_jsonl_file_prefix
    bool_start_tabby_server_on_modal = True if args.start_tabby_server_on_modal == "1" else False

    eval_code_completion(args.endpoint,
                         args.token,
                         args.model,
                         args.jsonl_file,
                         f"{output_jsonl_file_prefix}-prediction.jsonl",
                         f"{output_jsonl_file_prefix}-evaluation.jsonl",
                         f"{output_jsonl_file_prefix}-cross-file-content-prediction.jsonl",
                         f"{output_jsonl_file_prefix}-cross-file-content-evaluation.jsonl",
                         bool_start_tabby_server_on_modal)
