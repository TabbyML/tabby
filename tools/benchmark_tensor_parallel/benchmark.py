import ctranslate2
import argparse
import os
import collections
import time
import GPUtil
import sentencepiece as spm
import concurrent.futures

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


class BenchmarkResult(
    collections.namedtuple(
        "BenchmarkResult",
        (
                "generation_time",
                "num_tokens",
                "max_gpu_mem",
        ),
    )
):
    pass


def build_prompt(sp, inputs):
    prompt_tokens = []
    for question in inputs:
        input_tokens = ["<s>"] + sp.encode_as_pieces(
            f"{B_INST} {question.strip()} {E_INST}"
        )
        prompt_tokens.append(input_tokens)
    return prompt_tokens


def count_tokens(generated_token):
    count = 0
    for output in generated_token:
        count += len(output)
    return count


def avg_tokens(generated_token):
    return count_tokens(generated_token) / len(generated_token)


def process_prompt(generator, max_generation_length, generated_token, prompt):
    step_results = generator.generate_tokens(
        prompt,
        max_length=max_generation_length,
        sampling_temperature=0.6,
        sampling_topk=20,
        sampling_topp=1,
    )
    for step_result in step_results:
        batch_id = step_result.batch_id
        generated_token[batch_id].append(step_result.token)


def benchmark_generation(generator,
                         sp,
                         prompt_tokens,
                         generated_file,
                         mode,
                         batch_size):
    max_generation_length = 512
    generated_token = [[] for _ in range(len(prompt_tokens))]
    generated_text = ["" for _ in range(len(prompt_tokens))]
    tokens_buffer = []
    elapsed_time = None
    num_tokens = 0

    if mode == "sequence":
        start_all = time.time()
        for i in range(0, len(prompt_tokens), batch_size):
            step_results = generator.generate_tokens(
                prompt_tokens[i:i + batch_size],
                max_length=max_generation_length,
                sampling_temperature=0.6,
                sampling_topk=20,
                sampling_topp=1,
            )
            for step_result in step_results:
                batch_id = step_result.batch_id
                generated_token[batch_id].append(step_result.token)
        end_all = time.time()
        elapsed_time = end_all - start_all
        num_tokens = count_tokens(generated_token)
    elif mode == "parallel":
        nb_process = len(prompt_tokens) / batch_size + 1
        start_all = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=nb_process) as executor:
            futures = [executor.submit(process_prompt, generator, max_generation_length, generated_token,
                                       prompt_tokens[index:index + batch_size])
                       for index in range(0, len(prompt_tokens), batch_size)]
        num_tokens = count_tokens(generated_token)
        end_all = time.time()
        elapsed_time = end_all - start_all

    memory_gpus = float(GPUtil.getGPUs()[0].memoryUsed)

    # save answer to file
    for index in range(0, len(generated_token)):
        for token in generated_token[index]:
            is_new_word = token.startswith("▁")
            if is_new_word and tokens_buffer:
                word = sp.decode(tokens_buffer)
                if word:
                    if generated_text[index]:
                        word = ' ' + word
                    generated_text[index] += word
                tokens_buffer = []
            tokens_buffer.append(token)
        if tokens_buffer:
            word = sp.decode(tokens_buffer)
            if generated_text[index]:
                word = ' ' + word
            generated_text[index] += word
            tokens_buffer = []

    # write result to target file
    target_file = os.path.abspath(generated_file)
    if ctranslate2.MpiInfo.getCurRank() == 0:
        with open(target_file, 'w') as file:
            for index in range(len(generated_text)):
                file.write(f"answer{index}: ")
                file.write(generated_text[index])
                file.write(f"\n\n")

    return BenchmarkResult(elapsed_time, num_tokens, memory_gpus)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["sequence", "parallel"],
        default="sequence",
        help="benchmark in parallel or sequence mode",
    )
    parser.add_argument("--model_path", type=str, help="model path")
    parser.add_argument("--src", type=str, help="source file")
    parser.add_argument("--target", type=str, help="target file")
    parser.add_argument("--batch_size", type=int, help="batch size")
    args = parser.parse_args()

    print("Loading the model...")
    generator = ctranslate2.Generator(args.model_path, device="cuda", tensor_parallel=True, inter_threads=2)
    sp = spm.SentencePieceProcessor(os.path.join(args.model_path, "tokenizer.model"))

    if not os.path.exists(args.src):
        raise Exception("No source file found: " + args.src)
    # Open the file in read mode
    with open(args.src, 'r') as file:
        # Read all lines from the file and create a list
        inputs = file.readlines()

    prompt_tokens = build_prompt(sp, inputs)
    result = benchmark_generation(generator, sp, prompt_tokens, args.target, args.mode, args.batch_size)
    if ctranslate2.MpiInfo.getCurRank() == 0:
        print("Benchmark result (%d sample(s)):" % len(prompt_tokens))
        print("- Generation time: %.2f s" % result.generation_time)
        print("- Number of tokens: %d" % result.num_tokens)
        print("- Throughput: %.1f" % (result.num_tokens / result.generation_time))
        print("- max. GPU memory usage: %dMB" % int(result.max_gpu_mem))


if __name__ == "__main__":
    main()
