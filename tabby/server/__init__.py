import logging
import os

import uvicorn
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = os.environ.get("MODEL_NAME")
MODEL_BACKEND = os.environ.get("MODEL_BACKEND", "python")

if MODEL_BACKEND == "triton":
    model_backend = TritonService(
        tokenizer_name=MODEL_NAME,
        host=os.environ.get("TRITON_HOST", "triton"),
        port=os.environ.get("TRITON_PORT", "8001"),
    )
else:
    model_backend = PythonModelService(MODEL_NAME)


@app.post("/v1/completions")
async def completions(request: CompletionRequest) -> CompletionResponse:
    response = model_backend(request)
    events.log_completions(request, response)
    return response


@app.post("/v1/completions/{id}/choices/{index}/view")
async def view(id: str, index: int) -> JSONResponse:
    events.log_view(id, index)
    return JSONResponse(content="ok")


@app.post("/v1/completions/{id}/choices/{index}/selection")
async def selection(id: str, index: int) -> JSONResponse:
    events.log_selection(id, index)
    return JSONResponse(content="ok")
