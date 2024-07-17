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


def generate_predictions(endpoint: str, token: str, jsonl_file: str, output_prediction_jsonl_file: str):
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
        payload = {
            "language": row['language'],
            "segments": {
                "prefix": row['segments.prefix'],
                "suffix": row['segments.suffix']
            }
        }

        # TODO: Add parallelism support
        url = f"{endpoint}/v1/completions"
        response = send_request_with_retry(url, headers, payload, timeout=10, max_retries=10)

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

    df_success.to_json(output_prediction_jsonl_file, orient='records', lines=True)


def eval_code_completion(endpoint: str,
                         token: str,
                         model: str,
                         jsonl_file: str,
                         output_prediction_jsonl_file: str,
                         output_evaluation_jsonl_file: str,
                         need_manager_modal: bool):
    # Start modal tabby server
    process = None
    if need_manager_modal:
        process = start_tabby_server(model)

    # Check the service health
    logging.info("Checking service health...")
    check_service_health(endpoint, token)

    # Generate predictions
    logging.info("Generating predictions...")
    generate_predictions(endpoint, token, jsonl_file, output_prediction_jsonl_file)
    logging.info("Predictions generated!")

    # Run the evaluation
    logging.info("Running evaluation...")
    evaluation(output_prediction_jsonl_file, output_evaluation_jsonl_file)
    logging.info("Evaluation completed")

    # Stop the server
    if need_manager_modal and process:
        logging.info("Stopping server...")
        send_sigint_to_process(process)
        logging.info("Server stopped!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval tabby code completion.")
    parser.add_argument("--endpoint", type=str, required=True, help="tabby server endpoint.")
    parser.add_argument("--token", type=str, required=True, default="", help="tabby server token.")
    parser.add_argument("--model", type=str, required=True, help="evaluation model.")
    parser.add_argument("--jsonl_file", type=str, default="data.jsonl", help="evaluation jsonl file.")
    parser.add_argument("--output_prediction_jsonl_file", type=str, help="output prediction jsonl file.")
    parser.add_argument("--output_evaluation_jsonl_file", type=str, help="output evaluation jsonl file.")
    parser.add_argument("--need_manager_modal", type=str, default="1",
                        help="Whether a manager modal is needed. Accepts 1 or another.")

    args = parser.parse_args()
    bool_need_manager_modal = True if args.need_manager_modal == "1" else False
    eval_code_completion(args.endpoint,
                         args.token,
                         args.model,
                         args.jsonl_file,
                         args.output_prediction_jsonl_file,
                         args.output_evaluation_jsonl_file,
                         bool_need_manager_modal)
