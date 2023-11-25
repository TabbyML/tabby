import json
import os, subprocess
import modal

import requests
import pandas as pd

import base64
from io import BytesIO

import matplotlib.pyplot as plt

from typing import Dict, List

from dagster import (
    AssetExecutionContext,
    MetadataValue,
    asset,
    get_dagster_logger,
    op,
    StaticPartitionsDefinition,
    MultiPartitionsDefinition,
    AssetIn,
    Field,
    Int,
    file_relative_path
)
from . import analyze, create_csv
from dagstermill import define_dagstermill_asset


@asset(
    partitions_def=MultiPartitionsDefinition(
        {
            "model_id" : StaticPartitionsDefinition(['TabbyML/StarCoder-1B', 'TabbyML/StarCoder-3B', 'TabbyML/StarCoder-7B', 'TabbyML/WizardCoder-1B', 'TabbyML/WizardCoder-3B', 'TabbyML/CodeLlama-7B', 'TabbyML/CodeLlama-13B']),
            "language" : StaticPartitionsDefinition(["python", "java", "csharp", "typescript"]),
            
        }
    ))
def model_predict(context: AssetExecutionContext) -> None:
    model_id = context.partition_key.keys_by_dimension["model_id"]
    language = context.partition_key.keys_by_dimension["language"]

    my_env = os.environ.copy()
    my_env["MODEL_ID"] = model_id
    
    context.add_output_metadata(metadata={"model_id": MetadataValue.md(model_id)})

    files = 'line_completion.jsonl, line_completion_rg1_bm25.jsonl, line_completion_oracle_bm25.jsonl'

    p = subprocess.Popen(["modal", "run", "./modal/predict.py","--language", language, "--files", files], env=my_env)
    p.wait()
    context.add_output_metadata(metadata={'modal run': MetadataValue.md("success!")})


@asset(
    partitions_def=MultiPartitionsDefinition(
        {
            "model_id" : StaticPartitionsDefinition(['TabbyML/StarCoder-1B', 'TabbyML/StarCoder-3B', 'TabbyML/StarCoder-7B', 'TabbyML/WizardCoder-1B', 'TabbyML/WizardCoder-3B', 'TabbyML/CodeLlama-7B', 'TabbyML/CodeLlama-13B']),
            "language" : StaticPartitionsDefinition(["python", "java", "csharp", "typescript"]),       
        }
    ), deps=[model_predict])
def matching(context) -> None:
    model_id = context.partition_key.keys_by_dimension["model_id"]
    language = context.partition_key.keys_by_dimension["language"]
    

    model = model_id.split("/")[-1]
    for file in ["line_completion.jsonl", "line_completion_rg1_bm25.jsonl", "line_completion_oracle_bm25.jsonl"]:
        analyze.analyze(model, language, file)

@asset
def tabby_eval_result():
    create_csv.create_csv()



@asset(deps=[tabby_eval_result])
def tabby_dataset():
    return pd.read_csv(file_relative_path(__file__,'tabby.csv'))

tabby_jupyter_notebook = define_dagstermill_asset(
    name = 'tabby_jupyter',
    notebook_path = file_relative_path(__file__, "tabby_eval.ipynb"),
    ins={"df": AssetIn("tabby_dataset")},
)

