# Evaluating Code Completion Quality

## Introduction

This directory contains three Python scripts for evaluating code completion quality:

* `compute_code_completion.py`: Evaluates code completion quality using parameters.
* `compute_metrics.py`: Evaluates code completion quality given prediction / groundtruth
* `app.py`: A standalone Modal Tabby Serve service.

## Usage

Run the `compute_code_completion.py` script directly. Here’s an example:

```bash
python compute_code_completion.py \
  --endpoint https://moqimoqidea--tabby-server-app-serve-dev.modal.run \
  --token auth_a51a5e20bcd9478d83e4f26fb87055d1 \
  --model TabbyML/StarCoder-1B \
  --jsonl_file data.jsonl
```

This script will call the Tabby service and evaluate the quality of code completion. The script’s parameters are as follows:

```bash
python compute_code_completion.py -h
usage: compute_code_completion.py [-h] --endpoint ENDPOINT --token TOKEN --model MODEL
                               [--jsonl_file JSONL_FILE] [--need_manager_modal NEED_MANAGER_MODAL]

eval tabby code completion.

options:
  -h, --help            show this help message and exit
  --endpoint ENDPOINT   Tabby server endpoint.
  --token TOKEN         Tabby server token.
  --model MODEL         Evaluation model.
  --jsonl_file JSONL_FILE
                        Evaluation JSONL file.
  --need_manager_modal NEED_MANAGER_MODAL
                        Whether a manager modal is needed. Accepts 1 or another.
```

If you already have a Tabby service running, you can set the `need_manager_modal` parameter to 0 to avoid starting a standalone Tabby service. Example:

```bash
python compute_code_completion.py \
  --endpoint https://moqimoqidea--tabby-server-app-serve-dev.modal.run \
  --token auth_a51a5e20bcd9478d83e4f26fb87055d1 \
  --model TabbyML/StarCoder-1B \
  --jsonl_file data.jsonl \
  --need_manager_modal 0
```

If you have a JSONL file with code completion results, you can use the `compute_metrics.py` script. Example:

```bash
python compute_metrics.py --prediction_jsonl_file 20240714204945-TabbyML-StarCoder-1B.jsonl
```

The script’s parameters are as follows:

```bash
python compute_metrics.py -h
usage: compute_metrics.py [-h] [--prediction_jsonl_file PREDICTION_JSONL_FILE]

eval tabby code completion JSONL.

options:
  -h, --help            show this help message and exit
  --prediction_jsonl_file PREDICTION_JSONL_FILE
                        Prediction JSONL file.
```

Feel free to reach out if you have any questions or need further assistance!
