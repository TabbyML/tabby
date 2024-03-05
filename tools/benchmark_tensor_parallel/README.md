## Benchmark tools

This directory contains script to test the tensor parallelism mode.

### Requirements

* Python 3
* Following this [doc](../../docs/parallel.md#model-and-tensor-parallelism) to configure the environment.

```bash
python3 -m pip install -r requirements.txt
```

### Usage

```text
mpirun -np 2 -hostfile hostfile python3 benchmark.py --mode <MODE> --model_path <path> --src <input-file> --target <output-file> --batch_size <size>
```