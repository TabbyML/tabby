import time
from typing import List

import numpy as np
import tritonclient.grpc as client_util
from transformers import AutoTokenizer
from tritonclient.utils import InferenceServerException, np_to_triton_dtype

from ..models import Choice, CompletionRequest, CompletionResponse
from .language_presets import LanguagePresets
from .prompt_rewriter import PromptRewriteFailed, PromptRewriter
from .utils import random_completion_id, trim_with_stop_words


class TritonService:
    def __init__(
        self,
        tokenizer_name,
        host: str = "localhost",
        port: int = 8001,
        verbose: bool = False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.client = client_util.InferenceServerClient(
            url=f"{host}:{port}", verbose=verbose
        )
        self.rewriter = PromptRewriter()

    def generate(self, data: CompletionRequest) -> List[Choice]:
        n = 1
        np_type = np.uint32
        model_name = "fastertransformer"

        preset = LanguagePresets[data.language]

        try:
            prompt = self.rewriter.rewrite(preset, data.prompt)
        except PromptRewriteFailed:
            prompt = data.prompt
        except Exception as e:
            raise e

        input_start_ids = np.expand_dims(self.tokenizer.encode(prompt), 0)
        input_start_ids = np.repeat(input_start_ids, n, axis=0).astype(np_type)
        prompt_len = input_start_ids.shape[1]
        input_len = prompt_len * np.ones([input_start_ids.shape[0], 1]).astype(np_type)

        prompt_tokens: int = input_len[0][0]
        output_len = np.ones_like(input_len).astype(np_type) * preset.max_length

        stop_word_list = np.repeat(
            to_word_list_format([preset.stop_words], self.tokenizer),
            input_start_ids.shape[0],
            axis=0,
        )

        inputs = [
            prepare_tensor("input_ids", input_start_ids),
            prepare_tensor("input_lengths", input_len),
            prepare_tensor("request_output_len", output_len),
            prepare_tensor("stop_words_list", stop_word_list),
        ]

        result = self.client.infer(model_name, inputs)

        output_data = result.as_numpy("output_ids")
        if output_data is None:
            raise RuntimeError("No output data")

        output_data = output_data.squeeze(1)
        sequence_lengths = result.as_numpy("sequence_length").squeeze(1)
        gen_len = sequence_lengths - input_len.squeeze(1)

        decoded = [
            self.tokenizer.decode(out[prompt_len : prompt_len + g])
            for g, out in zip(gen_len, output_data)
        ]
        trimmed = [trim_with_stop_words(d, preset.stop_words) for d in decoded]
        return [Choice(index=i, text=text) for i, text in enumerate(trimmed)]

    def __call__(self, data: CompletionRequest) -> CompletionResponse:
        choices = self.generate(data)
        return CompletionResponse(
            id=random_completion_id(), created=int(time.time()), choices=choices
        )


def prepare_tensor(name: str, tensor_input):
    t = client_util.InferInput(
        name, tensor_input.shape, np_to_triton_dtype(tensor_input.dtype)
    )
    t.set_data_from_numpy(tensor_input)
    return t


def to_word_list_format(word_dict, tokenizer):
    flat_ids = []
    offsets = []
    for word_dict_item in word_dict:
        item_flat_ids = []
        item_offsets = []

        for word in word_dict_item:
            ids = tokenizer.encode(word)

            if len(ids) == 0:
                continue

            item_flat_ids += ids
            item_offsets.append(len(ids))

            if word == "\n\n":
                ids = tokenizer.encode("\n") * 2
                item_flat_ids += ids
                item_offsets.append(len(ids))

        flat_ids.append(np.array(item_flat_ids))
        offsets.append(np.cumsum(np.array(item_offsets)))

    pad_to = max(1, max(len(ids) for ids in flat_ids))

    for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
        flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
        offsets[i] = np.pad(offs, (0, pad_to - len(offs)), constant_values=-1)

    return np.array([flat_ids, offsets], dtype=np.int32).transpose((1, 0, 2))
