import argparse
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def transform(language: str,
              legacy_jsonl_file: str,
              output_data_jsonl_file: str):
    logging.info(f"Start transforming legacy data from {legacy_jsonl_file} to new format.")

    with open(legacy_jsonl_file, 'r') as source_file, open(output_data_jsonl_file, 'w') as destination_file:
        for line in source_file:
            source_data = json.loads(line)

            transformed_data = {
                "language": language,
                "segments": {
                    "prefix": source_data["prompt"],
                    "suffix": source_data["right_context"],
                    "crossfile_context": source_data["crossfile_context"]
                },
                "groundtruth": source_data["groundtruth"],
                "metadata": source_data["metadata"]
            }

            destination_file.write(json.dumps(transformed_data) + '\n')

    logging.info(f"End transforming legacy data to {output_data_jsonl_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="transform legacy jsonl file to new format.")
    parser.add_argument("--language", type=str, help="programming language.")
    parser.add_argument("--legacy_jsonl_file", type=str, help="legacy jsonl file.")
    parser.add_argument("--output_data_jsonl_file", type=str, help="output data jsonl file.")

    args = parser.parse_args()
    transform(args.language, args.legacy_jsonl_file, args.output_data_jsonl_file)
