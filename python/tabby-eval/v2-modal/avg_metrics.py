import argparse
import logging

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def avg_compute(evaluation_jsonl_file):
    df = pd.read_json(evaluation_jsonl_file, lines=True)

    avg_results = {
        "block_accuracy": df["block_accuracy"].mean(),
        "block_edit_distance": df["block_edit_distance"].mean(),
        "line_accuracy": df["line_accuracy"].mean(),
        "avg_line_edit_distance": df["avg_line_edit_distance"].mean(),
    }

    logging.info(f"Average results: {avg_results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="avg tabby code completion metrics.")
    parser.add_argument("--evaluation_jsonl_file", type=str, help="evaluation jsonl file.")

    args = parser.parse_args()
    avg_compute(args.evaluation_jsonl_file)
