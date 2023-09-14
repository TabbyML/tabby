import os
import sys

import ctranslate2
import sentencepiece as spm


def main():
    model_dir = sys.argv[1]
    system_prompt = sys.argv[2] if len(sys.argv) > 2 else None

    print("Loading the model...")
    generator = ctranslate2.Generator(model_dir, device="cuda")
    sp = spm.SentencePieceProcessor(os.path.join(model_dir, "tokenizer.model"))

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
            prompt_tokens = build_prompt(sp, dialog)
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
        print("Llama2: ", end="", flush=True)

        text_output = ""

        for word in generate_words(sp, step_results):
            if text_output:
                word = " " + word
            print(word, end="", flush=True)
            text_output += word

        print("")

        dialog.append({"role": "assistant", "content": text_output.strip()})


def generate_words(sp, step_results):
    tokens_buffer = []

    for step_result in step_results:
        is_new_word = step_result.token.startswith("▁")

        if is_new_word and tokens_buffer:
            word = sp.decode(tokens_buffer)
            if word:
                yield word
            tokens_buffer = []

        tokens_buffer.append(step_result.token_id)

    if tokens_buffer:
        word = sp.decode(tokens_buffer)
        if word:
            yield word


# The code below is adapted from
# https://github.com/facebookresearch/llama/blob/6c7fe276574e78057f917549435a2554000a876d/llama/generation.py#L225-L268

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def build_prompt(sp, dialog):
    if dialog[0]["role"] == "system":
        dialog = [
            {
                "role": dialog[1]["role"],
                "content": B_SYS + dialog[0]["content"] + E_SYS + dialog[1]["content"],
            }
        ] + dialog[2:]

    assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )

    dialog_tokens = sum(
        [
            ["<s>"]
            + sp.encode_as_pieces(
                f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} "
            )
            + ["</s>"]
            for prompt, answer in zip(
                dialog[::2],
                dialog[1::2],
            )
        ],
        [],
    )

    assert (
        dialog[-1]["role"] == "user"
    ), f"Last message must be from user, got {dialog[-1]['role']}"

    dialog_tokens += ["<s>"] + sp.encode_as_pieces(
        f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"
    )

    return dialog_tokens


if __name__ == "__main__":
    main()
