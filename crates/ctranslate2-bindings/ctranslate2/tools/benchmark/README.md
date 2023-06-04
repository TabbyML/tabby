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

The Docker image must contain 3 executable files at its root:

* `/tokenize $input $output`
* `/detokenize $input $output`
* `/translate $device $input $output`, where:
  * `$device` is "CPU" or "GPU"
  * `$input` is the path to the tokenized input file
  * `$output` is the path where the tokenized output should be written

The benchmark script will report multiple metrics. The results can be aggregated over multiple runs using the option `--num_samples N`. See `python3 benchmark.py -h` for additional options.

Note: the script focuses on raw decoding performance so the following steps are **not** included in the translation time:

* tokenization
* detokenization
* model initialization (obtained by translating an empty file)

### Reproducing the benchmark numbers from the README

We use the script `benchmark_all.py` to produce the benchmark numbers in the main [README](https://github.com/OpenNMT/CTranslate2#benchmarks). The script builds all Docker images defined in subdirectories and reports the results as a Markdown table. The execution can take up to 3 hours.

```text
# Run CPU benchmark:
python3 benchmark_all.py cpu

# Run GPU benchmark:
python3 benchmark_all.py gpu
```
