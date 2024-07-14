# Evaluating Code Completion Quality

## Introduction

This directory contains three Python scripts for evaluating code completion quality:

* `eval_code_completion.py`: Evaluates code completion quality using parameters.
* `eval_code_completion_jsonl.py`: Evaluates code completion quality using a results file.
* `app.py`: A standalone Modal Tabby Serve service.

## Usage

Run the `eval_code_completion.py` script directly. Here’s an example:

```bash
python eval_code_completion.py \
  --endpoint https://moqimoqidea--tabby-server-app-serve-dev.modal.run \
  --token auth_a51a5e20bcd9478d83e4f26fb87055d1 \
  --model TabbyML/StarCoder-1B \
  --jsonl_file data.jsonl
```

This script will call the Tabby service and evaluate the quality of code completion. The script’s parameters are as follows:

```bash
python eval_code_completion.py -h
usage: eval_code_completion.py [-h] --endpoint ENDPOINT --token TOKEN --model MODEL
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
python eval_code_completion.py \
  --endpoint https://moqimoqidea--tabby-server-app-serve-dev.modal.run \
  --token auth_a51a5e20bcd9478d83e4f26fb87055d1 \
  --model TabbyML/StarCoder-1B \
  --jsonl_file data.jsonl \
  --need_manager_modal 0
```

If you have a JSONL file with code completion results, you can use the `eval_code_completion_jsonl.py` script. Example:

```bash
python eval_code_completion_jsonl.py --prediction_jsonl_file 20240714204945-TabbyML-StarCoder-1B.jsonl
```

The script’s parameters are as follows:

```bash
python eval_code_completion_jsonl.py -h
usage: eval_code_completion_jsonl.py [-h] [--prediction_jsonl_file PREDICTION_JSONL_FILE]

eval tabby code completion JSONL.

options:
  -h, --help            show this help message and exit
  --prediction_jsonl_file PREDICTION_JSONL_FILE
                        Prediction JSONL file.
```

Feel free to reach out if you have any questions or need further assistance!
