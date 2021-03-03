import GPUtil
import docker
import os
import pyonmttok
import sacrebleu
import sys
import time
import argparse

client = docker.from_env()

docker_version = client.version()["Version"]
docker_version_numbers = docker_version.split(".")
docker_major_version = int(docker_version_numbers[0])
docker_minor_version = int(docker_version_numbers[1])

def get_bleu_score(hyp_file, ref_file, detokenize_fn=None):
    with open(hyp_file) as hyp, open(ref_file) as ref:
        if detokenize_fn is not None:
            hyp = map(detokenize_fn, hyp)
        bleu = sacrebleu.corpus_bleu(hyp, [ref])
        return bleu.score

def monitor_container(container, poll_interval=1, use_gpu=False):
    max_cpu_mem = 0
    max_gpu_mem = 0
    while True:
        try:
            container.wait(timeout=1)
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
    logs = container.logs()
    return logs.decode("utf-8"), max_cpu_mem, max_gpu_mem

class BenchmarkResult:
    def __init__(self, tokens_per_sec, bleu, max_cpu_mem, max_gpu_mem):
        self.tokens_per_sec = tokens_per_sec
        self.bleu = bleu
        self.max_cpu_mem = max_cpu_mem
        self.max_gpu_mem = max_gpu_mem
        self.weight = 1

    def combine(self, other_result):
        self.tokens_per_sec = (
            (self.weight * self.tokens_per_sec + other_result.weight * other_result.tokens_per_sec)
            / (self.weight + other_result.weight))
        self.max_cpu_mem = max(self.max_cpu_mem, other_result.max_cpu_mem)
        self.max_gpu_mem = max(self.max_gpu_mem, other_result.max_gpu_mem)
        self.weight += 1

def benchmark_image(image_name,
                    command,
                    source_file,
                    target_file,
                    tokenizer=None,
                    mount_dir="/data",
                    num_threads=4,
                    use_gpu=False):
    source_file = source_file if os.path.isabs(source_file) else os.path.abspath(source_file)
    target_file = target_file if os.path.isabs(target_file) else os.path.abspath(target_file)
    source_dir, source_file = os.path.split(source_file)
    target_dir, target_file = os.path.split(target_file)
    output_file = source_file + ".out"
    if source_dir != target_dir:
        raise ValueError("Source and target file should be in the same directory")
    data_dir = source_dir

    detokenize_fn = None
    if tokenizer is not None:
        detokenize_fn = lambda line: tokenizer.detokenize(line.rstrip().split())

    kwargs = {}
    environment = {"OMP_NUM_THREADS": str(args.num_threads)}
    if use_gpu:
        if docker_major_version < 19 or (docker_major_version == 19 and docker_minor_version < 3):
            kwargs["runtime"] = "nvidia"
        else:
            kwargs["device_requests"] = [
                docker.types.DeviceRequest(count=0, capabilities=[['gpu']])
            ]
    else:
        kwargs["cpuset_cpus"] = ",".join(map(str, range(num_threads)))
        environment["CUDA_VISIBLE_DEVICES"] = ""
    container = client.containers.run(
        image_name,
        command % (os.path.join(mount_dir, source_file), os.path.join(mount_dir, output_file)),
        detach=True,
        mounts=[docker.types.Mount(mount_dir, data_dir, type="bind")],
        environment=environment,
        **kwargs)
    try:
        logs, max_cpu_mem, max_gpu_mem = monitor_container(container, use_gpu=use_gpu)
        tokens_per_sec = float(logs.strip())
        bleu = get_bleu_score(output_file, target_file, detokenize_fn)
        return BenchmarkResult(tokens_per_sec, bleu, max_cpu_mem, max_gpu_mem)
    finally:
        container.remove(force=True)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--image", required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--command", required=True)
parser.add_argument("--src", required=True)
parser.add_argument("--tgt", required=True)
parser.add_argument("--sp_model", required=True)
parser.add_argument("--num_samples", type=int, default=3)
parser.add_argument("--num_threads", type=int, default=4)
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--format", choices=["csv", "markdown"], default="csv")
args = parser.parse_args()

src_tok = args.src + ".tok"
tokenizer = pyonmttok.Tokenizer("none", sp_model_path=args.sp_model)
tokenizer.tokenize_file(args.src, src_tok, num_threads=args.num_threads)

result = None
for _ in range(args.num_samples):
    sample_result = benchmark_image(
        args.image,
        args.command,
        src_tok,
        args.tgt,
        tokenizer,
        num_threads=args.num_threads,
        use_gpu=args.gpu,
    )
    if result is None:
        result = sample_result
    else:
        result.combine(sample_result)

if args.format == "csv":
    print(
        "%s;%.2f;%.1f;%d;%d"
        % (
            args.name,
            result.bleu,
            result.tokens_per_sec,
            int(result.max_cpu_mem),
            int(result.max_gpu_mem),
        )
    )
elif args.format == "markdown":
    print(
        "| %s | %.1f | %s | %.2f |"
        % (
            args.name,
            result.tokens_per_sec,
            "%d | %d" % (int(result.max_gpu_mem), int(result.max_cpu_mem))
            if args.gpu
            else str(int(result.max_cpu_mem)),
            result.bleu,
        )
    )
