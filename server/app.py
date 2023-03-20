import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from models import CompletionsRequest, CompletionsResponse

app = FastAPI(
    title="TabbyServer",
    description="TabbyServer is the backend for tabby, serving code completion requests from code editor / IDE.",
    docs_url="/",
)


@app.post("/v1/completions")
async def completions(data: CompletionsRequest) -> CompletionsResponse:
    return CompletionsResponse()


@app.post("/v1/completions/{id}/choices/{index}/select")
async def select(id: str, index: int):
    return JSONResponse(content=dict(status="ok"))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000)
