import modal
import json
import os
import subprocess
import pandas as pd

from dagster import (
    AssetExecutionContext,
    MetadataValue,
    asset,
    StaticPartitionsDefinition,
    MultiPartitionsDefinition,
)
from . import analyze


@asset
def baseline() -> str:
    return "line_completion.jsonl"

@asset
def bm25() -> str:
    return "line_completion_rg1_bm25.jsonl"

@asset
def oracle() -> str:
    return "line_completion_oracle_bm25.jsonl"

@asset(
    partitions_def=MultiPartitionsDefinition(
        {
            "model_id" : StaticPartitionsDefinition(['TabbyML/StarCoder-1B', 'TabbyML/StarCoder-3B', 'TabbyML/StarCoder-7B', 'TabbyML/WizardCoder-1B', 'TabbyML/WizardCoder-3B', 'TabbyML/CodeLlama-7B', 'TabbyML/CodeLlama-13B']),
            "language" : StaticPartitionsDefinition(["python", "java", "csharp", "typescript"]),
            
        }
    ))
def predict_baseline(context: AssetExecutionContext, baseline: str) -> None:
    model_id = context.partition_key.keys_by_dimension["model_id"]
    language = context.partition_key.keys_by_dimension["language"]

    my_env = os.environ.copy()
    my_env["MODEL_ID"] = model_id
    
    context.add_output_metadata(metadata={"model_id": MetadataValue.md(model_id)})

    files = baseline

    p = subprocess.Popen(["modal", "run", "./modal/predict.py","--language", language, "--files", files], env=my_env)
    p.wait()
    context.add_output_metadata(metadata={'modal run': MetadataValue.md("success!")})

@asset(
    partitions_def=MultiPartitionsDefinition(
        {
            "model_id" : StaticPartitionsDefinition(['TabbyML/StarCoder-1B', 'TabbyML/StarCoder-3B', 'TabbyML/StarCoder-7B', 'TabbyML/WizardCoder-1B', 'TabbyML/WizardCoder-3B', 'TabbyML/CodeLlama-7B', 'TabbyML/CodeLlama-13B']),
            "language" : StaticPartitionsDefinition(["python", "java", "csharp", "typescript"]),
            
        }
    ))
def predict_bm25(context: AssetExecutionContext, bm25: str) -> None:
    model_id = context.partition_key.keys_by_dimension["model_id"]
    language = context.partition_key.keys_by_dimension["language"]

    my_env = os.environ.copy()
    my_env["MODEL_ID"] = model_id
    
    context.add_output_metadata(metadata={"model_id": MetadataValue.md(model_id)})

    files = bm25

    p = subprocess.Popen(["modal", "run", "./modal/predict.py","--language", language, "--files", files], env=my_env)
    p.wait()
    context.add_output_metadata(metadata={'modal run': MetadataValue.md("success!")})


@asset(
    partitions_def=MultiPartitionsDefinition(
        {
            "model_id" : StaticPartitionsDefinition(['TabbyML/StarCoder-1B', 'TabbyML/StarCoder-3B', 'TabbyML/StarCoder-7B', 'TabbyML/WizardCoder-1B', 'TabbyML/WizardCoder-3B', 'TabbyML/CodeLlama-7B', 'TabbyML/CodeLlama-13B']),
            "language" : StaticPartitionsDefinition(["python", "java", "csharp", "typescript"]),
            
        }
    ))
def predict_oracle(context: AssetExecutionContext, oracle: str) -> None:
    model_id = context.partition_key.keys_by_dimension["model_id"]
    language = context.partition_key.keys_by_dimension["language"]

    my_env = os.environ.copy()
    my_env["MODEL_ID"] = model_id
    
    context.add_output_metadata(metadata={"model_id": MetadataValue.md(model_id)})

    files = oracle

    p = subprocess.Popen(["modal", "run", "./modal/predict.py","--language", language, "--files", files], env=my_env)
    p.wait()
    context.add_output_metadata(metadata={'modal run': MetadataValue.md("success!")})



@asset(
    partitions_def=MultiPartitionsDefinition(
        {
            "model_id" : StaticPartitionsDefinition(['TabbyML/StarCoder-1B', 'TabbyML/StarCoder-3B', 'TabbyML/StarCoder-7B', 'TabbyML/WizardCoder-1B', 'TabbyML/WizardCoder-3B', 'TabbyML/CodeLlama-7B', 'TabbyML/CodeLlama-13B']),
            "language" : StaticPartitionsDefinition(["python", "java", "csharp", "typescript"]),       
        }
    ), deps=[predict_baseline])
def matching_baseline(context) -> None:
    model_id = context.partition_key.keys_by_dimension["model_id"]
    language = context.partition_key.keys_by_dimension["language"]
    

    model = model_id.split("/")[-1]
    analyze.analyze(model, language, 'line_completion.jsonl')



@asset(
    partitions_def=MultiPartitionsDefinition(
        {
            "model_id" : StaticPartitionsDefinition(['TabbyML/StarCoder-1B', 'TabbyML/StarCoder-3B', 'TabbyML/StarCoder-7B', 'TabbyML/WizardCoder-1B', 'TabbyML/WizardCoder-3B', 'TabbyML/CodeLlama-7B', 'TabbyML/CodeLlama-13B']),
            "language" : StaticPartitionsDefinition(["python", "java", "csharp", "typescript"]),       
        }
    ), deps=[predict_bm25])
def matching_bm25(context) -> None:
    model_id = context.partition_key.keys_by_dimension["model_id"]
    language = context.partition_key.keys_by_dimension["language"]
    

    model = model_id.split("/")[-1]
    analyze.analyze(model, language, 'line_completion_rg1_bm25.jsonl')



@asset(
    partitions_def=MultiPartitionsDefinition(
        {
            "model_id" : StaticPartitionsDefinition(['TabbyML/StarCoder-1B', 'TabbyML/StarCoder-3B', 'TabbyML/StarCoder-7B', 'TabbyML/WizardCoder-1B', 'TabbyML/WizardCoder-3B', 'TabbyML/CodeLlama-7B', 'TabbyML/CodeLlama-13B']),
            "language" : StaticPartitionsDefinition(["python", "java", "csharp", "typescript"]),       
        }
    ), deps=[predict_oracle])
def matching_oracle(context) -> None:
    model_id = context.partition_key.keys_by_dimension["model_id"]
    language = context.partition_key.keys_by_dimension["language"]
    

    model = model_id.split("/")[-1]
    analyze.analyze(model, language, 'line_completion_oracle_bm25.jsonl')