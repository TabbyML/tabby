import logging
import os

import events
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from models import CompletionRequest, CompletionResponse
from triton import TritonService

app = FastAPI(
    title="TabbyServer",
    description="TabbyServer is the backend for tabby, serving code completion requests from code editor / IDE.",
    docs_url="/",
)

triton = TritonService(
    tokenizer_name=os.environ.get("TOKENIZER_NAME", None),
    host=os.environ.get("TRITON_HOST", "localhost"),
    port=os.environ.get("TRITON_PORT", "8001"),
)


@app.post("/v1/completions")
async def completions(request: CompletionRequest) -> CompletionResponse:
    response = triton(request)
    events.log_completions(request, response)
    return response


@app.post("/v1/completions/{id}/choices/{index}/selection")
async def selection(id: str, index: int) -> JSONResponse:
    events.log_selection(id, index)
    return JSONResponse(content="ok")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000)
