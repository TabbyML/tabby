import os
import sys

import ctranslate2
from transformers import AutoTokenizer


def main():
    model_dir = sys.argv[1]
    system_prompt = sys.argv[2] if len(sys.argv) > 2 else None

    print("Loading the model...")
    generator = ctranslate2.Generator(model_dir, device="cuda")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    context_length = 4096
    max_generation_length = 512
    max_prompt_length = context_length - max_generation_length

    dialog = []

    if system_prompt:
        dialog.append({"role": "system", "content": system_prompt})

    while True:
        print("")

        user_prompt = input("You: ")

        dialog.append({"role": "user", "content": user_prompt})

        while True:
            prompt_tokens = build_prompt(tokenizer, dialog)
            if len(prompt_tokens) <= max_prompt_length:
                break
            # Remove old conversations to reduce the prompt size.
            if system_prompt:
                dialog = [dialog[0]] + dialog[3:]
            else:
                dialog = dialog[2:]

        step_results = generator.generate_tokens(
            prompt_tokens,
            max_length=max_generation_length,
            sampling_temperature=0.6,
            sampling_topk=20,
            sampling_topp=1,
        )

        print("")
        print("Llama3: ", end="", flush=True)

        text_output = ""

        for word in generate_words(tokenizer, step_results):
            print(word, end="", flush=True)
            text_output += word

        print("")

        dialog.append({"role": "assistant", "content": text_output.strip()})


def generate_words(tokenizer, step_results):
    tokens_buffer = []

    for step_result in step_results:
        is_new_word = step_result.token.startswith("Ġ")

        if is_new_word and tokens_buffer:
            word = tokenizer.decode(tokens_buffer)
            if word:
                yield word
            tokens_buffer = []

        tokens_buffer.append(step_result.token_id)

    if tokens_buffer:
        word = tokenizer.decode(tokens_buffer)
        if word:
            yield word


B_ID, E_ID, E_INST = "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"


def build_prompt(tokenizer, dialog):
    begin_pos = 0
    if dialog[0]["role"] == "system":
        begin_pos = 1
    assert all([msg["role"] == "user" for msg in dialog[begin_pos::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[begin_pos + 1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )

    dialog_tokens = sum([
            tokenizer.tokenize(
                f"{B_ID} {(item['role'])} {E_ID} {(item['content']).strip()} {E_INST}"
                )
            for item in dialog
        ], [])
    dialog_tokens = ["<|begin_of_text|>"] + dialog_tokens + tokenizer.tokenize(
                f"{B_ID} assistant {E_ID}"
                )

    assert (
        dialog[-1]["role"] == "user"
    ), f"Last message must be from user, got {dialog[-1]['role']}"

    return dialog_tokens


if __name__ == "__main__":
    main()
