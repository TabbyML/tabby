import logging
import os

import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

from . import events
from .models import CompletionRequest, CompletionResponse
from .python import PythonModelService
from .triton import TritonService

app = FastAPI(
    title="TabbyServer",
    description="TabbyServer is the backend for tabby, serving code completion requests from code editor / IDE.",
    docs_url="/",
)

MODEL_BACKEND = os.environ.get("MODEL_BACKEND", "python")

if MODEL_BACKEND == "triton":
    model_backend = TritonService(
        tokenizer_name=os.environ.get("TRITON_TOKENIZER_NAME", None),
        host=os.environ.get("TRITON_HOST", "triton"),
        port=os.environ.get("TRITON_PORT", "8001"),
    )
else:
    model_backend = PythonModelService(os.environ["PYTHON_MODEL_NAME"])


@app.post("/v1/completions")
async def completions(request: CompletionRequest) -> CompletionResponse:
    response = model_backend(request)
    events.log_completions(request, response)
    return response


@app.post("/v1/completions/{id}/choices/{index}/selection")
async def selection(id: str, index: int) -> JSONResponse:
    events.log_selection(id, index)
    return JSONResponse(content="ok")
