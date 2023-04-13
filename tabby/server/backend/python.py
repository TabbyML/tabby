import time
from typing import List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from ..models import Choice, CompletionRequest, CompletionResponse
from .language_presets import LanguagePresets
from .utils import random_completion_id, trim_with_stop_words


class PythonModelService:
    def __init__(self, model_name, quantize=False):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        else:
            if quantize:
                raise ValueError("quantization on CPU is not implemented")

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
                load_in_8bit=quantize,
            )
            .to(device)
            .eval()
        )

    def generate(self, request: CompletionRequest) -> List[Choice]:
        preset = LanguagePresets.get(request.language, None)
        if preset is None:
            return []

        input_ids = self.tokenizer.encode(request.prompt, return_tensors="pt").to(
            self.device
        )
        res = self.model.generate(
            input_ids,
            max_length=preset.max_length,
        )
        output_ids = res[0][len(input_ids[0]) :]
        text = trim_with_stop_words(
            self.tokenizer.decode(output_ids), preset.stop_words
        )
        return [Choice(index=0, text=text)]

    def __call__(self, request: CompletionRequest) -> CompletionResponse:
        choices = self.generate(request)
        return CompletionResponse(
            id=random_completion_id(), created=int(time.time()), choices=choices
        )
