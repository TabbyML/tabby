import GPUtil
import argparse
import collections
import docker
import os
import sacrebleu
import tempfile
import time

client = docker.from_env()

docker_version = client.version()["Version"]
docker_version_numbers = docker_version.split(".")
docker_major_version = int(docker_version_numbers[0])
docker_minor_version = int(docker_version_numbers[1])


def _get_bleu_score(hyp_file, ref_file):
    with open(hyp_file) as hyp, open(ref_file) as ref:
        bleu = sacrebleu.corpus_bleu(hyp, [ref], force=True)
        return bleu.score


def _count_tokens(path):
    with open(path) as file:
        num_tokens = 0
        for line in file:
            num_tokens += len(line.strip().split(" "))
        return num_tokens


def _monitor_container(container, poll_interval=1, use_gpu=False):
    max_cpu_mem = 0
    max_gpu_mem = 0
    result = None

    while True:
        try:
            result = container.wait(timeout=1)
            break
        except:
            pass
        stats = container.stats(stream=False)
        memory_stats = stats["memory_stats"]
        memory_usage = memory_stats.get("usage")
        if memory_usage is not None:
            max_cpu_mem = max(max_cpu_mem, float(memory_usage / 1000000))
        if use_gpu:
            max_gpu_mem = max(max_gpu_mem, float(GPUtil.getGPUs()[0].memoryUsed))

    if result is not None and result["StatusCode"] != 0:
        stderr = container.logs(stdout=False).decode("utf-8")
        raise RuntimeError(
            "Container exited with status code %d:\n\n%s"
            % (result["StatusCode"], stderr)
        )

    return max_cpu_mem, max_gpu_mem


def _process_file(image_name, script, input_file, output_file):
    input_dir = "/input"
    output_dir = "/output"
    client.containers.run(
        image_name,
        command=[
            os.path.join(input_dir, os.path.basename(input_file)),
            os.path.join(output_dir, os.path.basename(output_file)),
        ],
        entrypoint=script,
        remove=True,
        mounts=[
            docker.types.Mount(input_dir, os.path.dirname(input_file), type="bind"),
            docker.types.Mount(output_dir, os.path.dirname(output_file), type="bind"),
        ],
    )


def _tokenize(image_name, input_file, output_file):
    return _process_file(image_name, "/tokenize", input_file, output_file)


def _detokenize(image_name, input_file, output_file):
    return _process_file(image_name, "/detokenize", input_file, output_file)


def _start_translation(
    image_name,
    source_file,
    output_file,
    environment,
    num_cpus,
    use_gpu,
):
    kwargs = {}
    environment = environment.copy() if environment else {}
    environment["OMP_NUM_THREADS"] = str(num_cpus)

    if use_gpu:
        device = "GPU"
        if docker_major_version < 19 or (
            docker_major_version == 19 and docker_minor_version < 3
        ):
            kwargs["runtime"] = "nvidia"
        else:
            kwargs["device_requests"] = [
                docker.types.DeviceRequest(count=0, capabilities=[["gpu"]])
            ]
    else:
        device = "CPU"
        environment["CUDA_VISIBLE_DEVICES"] = ""

    data_dir = "/data"
    output_dir = "/output"
    container = client.containers.run(
        image_name,
        [
            device,
            os.path.join(data_dir, os.path.basename(source_file)),
            os.path.join(output_dir, os.path.basename(output_file)),
        ],
        entrypoint="/translate",
        detach=True,
        mounts=[
            docker.types.Mount(data_dir, os.path.dirname(source_file), type="bind"),
            docker.types.Mount(output_dir, os.path.dirname(output_file), type="bind"),
        ],
        environment=environment,
        **kwargs
    )

    return container


def _benchmark_translation(
    image_name,
    source_file,
    target_file,
    environment,
    num_cpus,
    use_gpu,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        source_file_tok = os.path.join(tmp_dir, "source.txt.tok")
        output_file_tok = os.path.join(tmp_dir, "output.txt.tok")
        output_file = os.path.join(tmp_dir, "output.txt")

        _tokenize(image_name, source_file, source_file_tok)

        container = _start_translation(
            image_name,
            source_file_tok,
            output_file_tok,
            environment,
            num_cpus,
            use_gpu,
        )

        try:
            start = time.time()
            max_cpu_mem, max_gpu_mem = _monitor_container(container, use_gpu=use_gpu)
            end = time.time()
            elapsed_time = end - start

            num_tokens = _count_tokens(output_file_tok)
            _detokenize(image_name, output_file_tok, output_file)
            bleu = _get_bleu_score(output_file, target_file)

            return elapsed_time, num_tokens, max_cpu_mem, max_gpu_mem, bleu
        finally:
            container.remove(force=True)


class BenchmarkResult(
    collections.namedtuple(
        "BenchmarkResult",
        (
            "total_time",
            "translation_time",
            "num_tokens",
            "max_cpu_mem",
            "max_gpu_mem",
            "bleu_score",
        ),
    )
):
    pass


def benchmark_image(
    image_name,
    source_file,
    target_file,
    num_samples=1,
    environment=None,
    num_cpus=4,
    use_gpu=False,
):
    source_file = os.path.abspath(source_file)
    target_file = os.path.abspath(target_file)

    initialization_time = None
    with tempfile.NamedTemporaryFile() as tmp_file:
        for _ in range(num_samples):
            container = _start_translation(
                image_name,
                tmp_file.name,
                tmp_file.name,
                environment,
                num_cpus,
                use_gpu,
            )
            try:
                start = time.time()
                container.wait()
                end = time.time()
                elapsed_time = end - start
                initialization_time = (
                    elapsed_time
                    if initialization_time is None
                    else min(initialization_time, elapsed_time)
                )
            finally:
                container.remove(force=True)

    total_time = None
    num_tokens = 0
    bleu = 0
    max_cpu_mem = 0
    max_gpu_mem = 0

    for _ in range(num_samples):
        results = _benchmark_translation(
            image_name,
            source_file,
            target_file,
            environment,
            num_cpus,
            use_gpu,
        )

        total_time = results[0] if total_time is None else min(total_time, results[0])
        num_tokens = results[1]
        max_cpu_mem = max(max_cpu_mem, results[2])
        max_gpu_mem = max(max_gpu_mem, results[3])
        bleu = results[4]

    translation_time = total_time - initialization_time
    return BenchmarkResult(
        total_time,
        translation_time,
        num_tokens,
        max_cpu_mem,
        max_gpu_mem,
        bleu,
    )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="aggregate results over this number of runs",
    )
    parser.add_argument("--num_cpus", type=int, default=4, help="number of CPUs to use")
    parser.add_argument("--gpu", action="store_true", help="run on GPU")
    parser.add_argument(
        "--env",
        type=str,
        nargs=2,
        action="append",
        default=[],
        help="add this environment variable to the Docker container",
    )
    parser.add_argument("image", type=str, help="name of Docker image to benchmark")
    parser.add_argument("src", type=str, help="source file")
    parser.add_argument("ref", type=str, help="reference file")
    args = parser.parse_args()

    result = benchmark_image(
        args.image,
        args.src,
        args.ref,
        num_samples=args.num_samples,
        environment={key: value for key, value in args.env},
        num_cpus=args.num_cpus,
        use_gpu=args.gpu,
    )

    print("Benchmark result (%d sample(s)):" % args.num_samples)
    print("- total time: %.2f s" % result.total_time)
    print("- translation time: %.2f s" % result.translation_time)
    print("- tokens per second: %.1f" % (result.num_tokens / result.translation_time))
    print("- max. CPU memory usage: %dMB" % int(result.max_cpu_mem))
    if args.gpu:
        print("- max. GPU memory usage: %dMB" % int(result.max_gpu_mem))
    print("- BLEU score: %.2f" % result.bleu_score)


if __name__ == "__main__":
    main()
