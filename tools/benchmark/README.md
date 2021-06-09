## Benchmark tools

This directory contains some scripts to benchmark translation systems.

### Requirements

* Python 3
* Docker

```bash
python3 -m pip install -r requirements.txt
```

### Usage

```text
python3 benchmark.py <IMAGE> <SOURCE> <REFERENCE>
```

The Docker image must contain 3 scripts at its root:

* `/tokenize.sh $input $output`
* `/detokenize.sh $input $output`
* `/translate.sh $device $input $output`, where:
  * `$device` is "CPU" or "GPU"
  * `$input` is the path to the tokenized input file
  * `$output` is the path where the tokenized output should be written

The benchmark script will report multiple metrics. The results can be aggregated over multiple runs using the option `--num_samples N`. See `python3 benchmark.py -h` for additional options.

Note: the script focuses on raw decoding performance so the following steps are **not** included in the translation time:

* tokenization
* detokenization
* model initialization (obtained by translating an empty file)

### Reproducing the benchmark numbers from the README

We use the script `benchmark_pretrained.py` to produce the benchmark numbers in the main [README](https://github.com/OpenNMT/CTranslate2#benchmarks). The directory `pretrained_transformer_base` contains the Docker images corresponding to the pretrained OpenNMT Transformers.

```text
# Run CPU benchmark:
python3 benchmark_pretrained.py cpu

# Run GPU benchmark:
python3 benchmark_pretrained.py gpu
```
