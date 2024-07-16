import argparse
import logging

import pandas as pd
from Levenshtein import ratio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def evaluation(prediction_jsonl_file):
    df = pd.read_json(prediction_jsonl_file, lines=True)
    total_score = 0

    for index, row in df.iterrows():
        groundtruth = row['groundtruth'].split('\n')
        prediction = row['prediction'].split('\n')

        # Get the first non-empty line
        groundtruth_first_non_empty = next((s for s in groundtruth if s.strip()), "")
        prediction_first_non_empty = next((s for s in prediction if s.strip()), "")

        # Calculate the ratio between the two
        score = ratio(groundtruth_first_non_empty, prediction_first_non_empty)

        # Add the score to the total score
        total_score += score

    # Calculate the average score
    average_score = total_score / len(df) if len(df) > 0 else 0
    logging.info(f"Evaluation result: file {prediction_jsonl_file}, score: {average_score}")

    return average_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval tabby code completion jsonl.")
    parser.add_argument("--prediction_jsonl_file", type=str, help="prediction jsonl file.")

    args = parser.parse_args()
    evaluation(args.prediction_jsonl_file)
