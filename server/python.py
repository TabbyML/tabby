import random
import string
import time
from typing import List

from models import Choice, CompletionRequest, CompletionResponse
from transformers import AutoModelForCausalLM, AutoTokenizer


class PythonModelService:
    def __init__(
        self,
        model_name,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, request: CompletionRequest) -> List[Choice]:
        input_ids = self.tokenizer.encode(request.prompt, return_tensors="pt")
        res = self.model.generate(input_ids, max_length=64)
        output_ids = res[0][len(input_ids[0]) :]
        text = self.tokenizer.decode(output_ids)
        return [Choice(index=0, text=text)]

    def __call__(self, request: CompletionRequest) -> CompletionResponse:
        choices = self.generate(request)
        return CompletionResponse(
            id=random_completion_id(), created=int(time.time()), choices=choices
        )


def random_completion_id():
    return "cmpl-" + "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(29)
    )
