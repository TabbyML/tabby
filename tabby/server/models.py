from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Choice(BaseModel):
    index: int
    text: str


class CompletionRequest(BaseModel):
    prompt: str = Field(
        example="def binarySearch(arr, left, right, x):\n    mid = (left +",
        description="The context to generate completions for, encoded as a string.",
    )


class CompletionResponse(BaseModel):
    id: str
    created: int
    choices: List[Choice]


class Event(BaseModel):
    type: str


class CompletionEvent(Event):
    id: str
    prompt: str
    created: int
    choices: List[Choice]

    @classmethod
    def build(cls, request: CompletionRequest, response: CompletionResponse):
        return cls(
            type="completion",
            id=response.id,
            prompt=request.prompt,
            created=response.created,
            choices=response.choices,
        )


class ChoiceEvent(Event):
    completion_id: str
    choice_index: int

    @classmethod
    def build_view(cls, id, index):
        return cls(type="view", completion_id=id, choice_index=index)

    @classmethod
    def build_select(cls, id, index):
        return cls(type="select", completion_id=id, choice_index=index)
