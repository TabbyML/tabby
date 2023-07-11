import sys
import argparse
import pandas as pd
import logging

from tabby_client import Client
from tabby_client.api.v1 import health
from tabby_client.api.v1 import completion

from tabby_client.models import CompletionRequest, CompletionRequest, Segments, Choice

import processing
import editdistance
import random


def valid_item(item: processing.Item):
    count_body_lines = len(item.body.splitlines())

    if count_body_lines > 10:
        return False

    return True


def scorer(label, prediction):
    distance = editdistance.eval(label, prediction)
    return max(0.0, 1.0 - distance / len(label))


def run_eval(args):
    api = "http://localhost:8080"
    client = Client(base_url=api, timeout=50)
    try:
        health.sync(client=client)
    except:
        print(f"Tabby Server is not ready, please check if '{api}' is correct.")
        return
    
    items = [x for x in processing.items_from_filepattern(args.filepattern) if valid_item(x)];
    if len(items) > args.max_records:
        random.seed(0xbadbeef)
        items = random.sample(items, args.max_records)
    

    for item in items:
        if not valid_item(item):
            continue

        request = CompletionRequest(
            language=item.language, segments=Segments(prefix=item.prefix)
        )

        resp: CompletionResponse = completion.sync(client=client, json_body=request)
        label = item.body
        prediction = resp.choices[0].text

        block_score = scorer(label, prediction)
        
        label_lines = label.splitlines()
        prediction_lines = prediction.splitlines()
        
        if len(label_lines) > 0 and len(prediction_lines) > 0:
            line_score = scorer(label_lines[0], prediction_lines[0])

        yield dict(
            prompt=item.prefix,
            prediction=prediction,
            label=label,
            block_score=block_score,
            line_score=line_score,
        )

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    parser = argparse.ArgumentParser(description='SxS eval for tabby')
    parser.add_argument('filepattern', type=str, help='File pattern to dataset.')
    parser.add_argument('max_records', type=int, help='Max number of records to be evaluated.')
    args = parser.parse_args()
    logging.info("args %s", args)
    df = pd.DataFrame(run_eval(args))
    print(df.to_json(orient='records', lines=True))
