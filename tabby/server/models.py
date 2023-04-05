from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class Choice(BaseModel):
    index: int
    text: str

# https://code.visualstudio.com/docs/languages/identifiers
class Language(str, Enum):
    UNKNOWN = "unknown"
    PYTHON = "python"
    JAVASCRIPT = "javascript"


class CompletionRequest(BaseModel):
    language: Language = Field(
        example=Language.PYTHON,
        default=Language.UNKNOWN,
        description="Language for completion request",
    )

    prompt: str = Field(
        example="def binarySearch(arr, left, right, x):\n    mid = (left +",
        description="The context to generate completions for, encoded as a string.",
    )


class CompletionResponse(BaseModel):
    id: str
    created: int
    choices: List[Choice]


class EventType(str, Enum):
    COMPLETION = "completion"
    VIEW = "view"
    SELECT = "select"


class Event(BaseModel):
    type: EventType


class CompletionEvent(Event):
    id: str
    language: Language
    prompt: str
    created: int
    choices: List[Choice]

    @classmethod
    def build(cls, request: CompletionRequest, response: CompletionResponse):
        return cls(
            type=EventType.COMPLETION,
            id=response.id,
            language=request.language,
            prompt=request.prompt,
            created=response.created,
            choices=response.choices,
        )


class ChoiceEvent(Event):
    completion_id: str
    choice_index: int


EventTypeMapping = {
    EventType.COMPLETION: CompletionEvent,
    EventType.VIEW: ChoiceEvent,
    EventType.SELECT: ChoiceEvent,
}
