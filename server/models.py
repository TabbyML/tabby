from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Choice(BaseModel):
    index: int
    text: str


class CompletionsRequest(BaseModel):
    prompt: str = Field(
        example="def binarySearch(arr, left, right, x):\n    mid = (left +",
        description="The context to generate completions for, encoded as a string.",
    )


class CompletionsResponse(BaseModel):
    id: str
    created: int
    choices: List[Choice]
