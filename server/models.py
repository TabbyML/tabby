from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Choice(BaseModel):
    index: int
    text: str


class CompletionsRequest(BaseModel):
    prompt: str = Field(
        example="def fib(n):",
        description="The context to generate completions for, encoded as a string.",
    )
    suffix: Optional[str] = Field(
        description="The suffix that comes after a completion of inserted code."
    )


class CompletionsResponse(BaseModel):
    id: str
    choices: List[Choice]
