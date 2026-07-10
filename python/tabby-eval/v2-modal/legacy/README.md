# Transform legacy data to new format

## Usage

Run the `transform_legacy_data.py` script directly. Here’s an example:

```bash
python transform_legacy_data.py \
  --language csharp \
  --legacy_jsonl_file head10_line_completion_rg1_bm25.jsonl \
  --output_data_jsonl_file data-with-crossfile-context.jsonl
```

This script will transform the legacy data to new format. The script’s parameters are as follows:

```bash
python transform_legacy_data.py -h
usage: transform_legacy_data.py [-h] [--language LANGUAGE] [--legacy_jsonl_file LEGACY_JSONL_FILE]
                                [--output_data_jsonl_file OUTPUT_DATA_JSONL_FILE]

transform legacy jsonl file to new format.

options:
  -h, --help            show this help message and exit
  --language LANGUAGE   programming language.
  --legacy_jsonl_file LEGACY_JSONL_FILE
                        legacy jsonl file.
  --output_data_jsonl_file OUTPUT_DATA_JSONL_FILE
                        output data jsonl file.
```

Feel free to reach out if you have any questions or need further assistance!
