import random
import string
import time
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .models import Choice, CompletionRequest, CompletionResponse


class PythonModelService:
    def __init__(
        self,
        model_name,
    ):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        else:
            device = torch.device("cpu")
            dtype = torch.float32

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=True
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                local_files_only=True,
            )
            .to(device)
            .eval()
        )

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
