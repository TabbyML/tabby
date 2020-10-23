## Benchmark tools

This directory contains some scripts to benchmark CTranslate2 and compare against OpenNMT-py and OpenNMT-tf.

### Requirements

* Python 3
* Docker

### Usage

```bash
pip install -r requirements.txt

# Run CPU benchmark:
./run.sh

# Run GPU benchmark:
./run.sh 1
```

The script outputs one result per line where each line consists of 5 fields separated by a semicolon:

1. Run name
1. BLEU score
1. Tokens per second
1. System maximum memory usage (MB)
1. GPU maximum memory usage (MB)
