import os

import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from models import CompletionsRequest, CompletionsResponse
from triton import TritonService

app = FastAPI(
    title="TabbyServer",
    description="TabbyServer is the backend for tabby, serving code completion requests from code editor / IDE.",
    docs_url="/",
)

triton = TritonService(os.environ["TOKENIZER_NAME"])


@app.post("/v1/completions")
async def completions(data: CompletionsRequest) -> CompletionsResponse:
    return triton(data)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000)
