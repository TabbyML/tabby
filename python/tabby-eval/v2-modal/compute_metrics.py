import argparse
import logging

import pandas as pd
from Levenshtein import ratio

from avg_metrics import avg_compute

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def evaluation(prediction_jsonl_file, output_evaluation_jsonl_file):
    df = pd.read_json(prediction_jsonl_file, lines=True)
    results = []

    for index, row in df.iterrows():
        groundtruth = row['groundtruth'].split('\n')
        prediction = row['prediction'].split('\n')

        line_accuracy = sum(1 for gt, pred in zip(groundtruth, prediction) if gt == pred) / len(
            groundtruth) if groundtruth else 0

        block_accuracy = 1 if '\n'.join(groundtruth) == '\n'.join(prediction) else 0

        line_edit_distances = [ratio(gt, pred) for gt, pred in zip(groundtruth, prediction)]
        avg_line_edit_distance = sum(line_edit_distances) / len(line_edit_distances) if line_edit_distances else 0

        block_edit_distance = ratio('\n'.join(groundtruth), '\n'.join(prediction))

        results.append({
            "line_accuracy": line_accuracy,
            "block_accuracy": block_accuracy,
            "avg_line_edit_distance": avg_line_edit_distance,
            "block_edit_distance": block_edit_distance
        })

    df = pd.concat([df, pd.DataFrame(results)], axis=1)

    df.to_json(output_evaluation_jsonl_file, orient='records', lines=True, force_ascii=False)
    logging.info(f"Evaluation result written to {output_evaluation_jsonl_file}")

    avg_compute(output_evaluation_jsonl_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval tabby code completion jsonl.")
    parser.add_argument("--prediction_jsonl_file", type=str, help="prediction jsonl file.")
    parser.add_argument("--output_evaluation_jsonl_file", type=str, help="output evaluation jsonl file.")

    args = parser.parse_args()
    evaluation(args.prediction_jsonl_file, args.output_evaluation_jsonl_file)
