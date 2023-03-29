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
from .utils import random_completion_id, trim_with_stopwords


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
        self.stopping_criteria_mappings = {}

    def generate(self, request: CompletionRequest) -> List[Choice]:
        # FIXME(meng): read preset from request.
        preset_name = "python"

        preset = LanguagePresets[preset_name]
        stopping_criteria_list = self.stopping_criteria_for_preset(preset_name)

        input_ids = self.tokenizer.encode(request.prompt, return_tensors="pt").to(
            self.device
        )
        res = self.model.generate(
            input_ids,
            max_length=preset.max_length,
            stopping_criteria=stopping_criteria_list,
        )
        output_ids = res[0][len(input_ids[0]) :]
        text = trim_with_stopwords(self.tokenizer.decode(output_ids), preset.stop_words)
        return [Choice(index=0, text=text)]

    def stopping_criteria_for_preset(self, name: str) -> StoppingCriteriaList:
        lst = self.stopping_criteria_mappings.get(name, None)
        if not lst:
            lst = self.stopping_criteria_mappings[name] = StoppingCriteriaList(
                [
                    StopWordsIdsCriteria(
                        [
                            self.tokenizer.encode(x)
                            for x in LanguagePresets[name].stop_words
                        ]
                    )
                ]
            )
        return self.stopping_criteria_mappings[name]

    def __call__(self, request: CompletionRequest) -> CompletionResponse:
        choices = self.generate(request)
        return CompletionResponse(
            id=random_completion_id(), created=int(time.time()), choices=choices
        )


class StopWordsIdsCriteria(StoppingCriteria):
    def __init__(self, stop_words_ids: List[str]):
        self.stop_words_ids = stop_words_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if len(input_ids) != 1:
            raise ValueError("Only 1-length list is handled")

        # FIXME(meng): trie based lookup.
        tokens = input_ids[0]
        for stop_word in self.stop_words_ids:
            if len(tokens) < len(stop_word):
                continue

            matched = True
            for i in range(len(stop_word)):
                if tokens[i - len(stop_word)] != stop_word[i]:
                    matched = False
                    break

            if matched:
                return True

        return False
