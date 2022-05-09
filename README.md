[![CI](https://github.com/OpenNMT/CTranslate2/workflows/CI/badge.svg)](https://github.com/OpenNMT/CTranslate2/actions?query=workflow%3ACI) [![PyPI version](https://badge.fury.io/py/ctranslate2.svg)](https://badge.fury.io/py/ctranslate2) [![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://opennmt.net/CTranslate2/) [![Gitter](https://badges.gitter.im/OpenNMT/CTranslate2.svg)](https://gitter.im/OpenNMT/CTranslate2?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge) [![Forum](https://img.shields.io/discourse/status?server=https%3A%2F%2Fforum.opennmt.net%2F)](https://forum.opennmt.net/)

# CTranslate2

CTranslate2 is a C++ and Python library for efficient inference with Transformer models. The project implements a custom runtime that applies many performance optimization techniques such as weights quantization, layers fusion, batch reordering, etc., to accelerate and reduce the memory usage of Transformer models on CPU and GPU. The following model types are currently supported:

* Encoder-decoder models: Transformer base/big, M2M-100, BART, mBART
* Decoder-only models: GPT-2

Compatible models should be first converted into an optimized model format. The library includes converters for multiple frameworks:

* [OpenNMT-py](https://opennmt.net/CTranslate2/guides/opennmt_py.html)
* [OpenNMT-tf](https://opennmt.net/CTranslate2/guides/opennmt_tf.html)
* [Fairseq](https://opennmt.net/CTranslate2/guides/fairseq.html)
* [Marian](https://opennmt.net/CTranslate2/guides/marian.html)
* [Transformers](https://opennmt.net/CTranslate2/guides/transformers.html)

The project is production-oriented and comes with [backward compatibility guarantees](https://opennmt.net/CTranslate2/versioning.html), but it also includes experimental features related to model compression and inference acceleration.

## Key features

* **Fast and efficient execution on CPU and GPU**<br/>The execution [is significantly faster and requires less resources](#benchmarks) than general-purpose deep learning frameworks on supported models and tasks thanks to many advanced optimizations: layer fusion, padding removal, batch reordering, in-place operations, caching mechanism, etc.
* **Quantization and reduced precision**<br/>The model serialization and computation support weights with [reduced precision](https://opennmt.net/CTranslate2/quantization.html): 16-bit floating points (FP16), 16-bit integers (INT16), and 8-bit integers (INT8).
* **Multiple CPU architectures support**<br/>The project supports x86-64 and AArch64/ARM64 processors and integrates multiple backends that are optimized for these platforms: [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html), [oneDNN](https://github.com/oneapi-src/oneDNN), [OpenBLAS](https://www.openblas.net/), [Ruy](https://github.com/google/ruy), and [Apple Accelerate](https://developer.apple.com/documentation/accelerate).
* **Automatic CPU detection and code dispatch**<br/>One binary can include multiple backends (e.g. Intel MKL and oneDNN) and instruction set architectures (e.g. AVX, AVX2) that are automatically selected at runtime based on the CPU information.
* **Parallel and asynchronous execution**<br/>Multiple batches can be processed in parallel and asynchronously using multiple GPUs or CPU cores.
* **Dynamic memory usage**<br/>The memory usage changes dynamically depending on the request size while still meeting performance requirements thanks to caching allocators on both CPU and GPU.
* **Lightweight on disk**<br/>Quantization can make the models 4 times smaller on disk with minimal accuracy loss. A full featured Docker image supporting GPU and CPU requires less than 500MB (with CUDA 10.0).
* **Simple integration**<br/>The project has few dependencies and exposes simple APIs in [Python](https://opennmt.net/CTranslate2/python/overview.html) and C++ to cover most integration needs.
* **Configurable and interactive decoding**<br/>[Advanced decoding features](https://opennmt.net/CTranslate2/decoding.html) allow autocompleting a partial sequence and returning alternatives at a specific location in the sequence.

Some of these features are difficult to achieve with standard deep learning frameworks and are the motivation for this project.

## Installation and usage

CTranslate2 can be installed with pip:

```bash
pip install ctranslate2
```

The Python module is used to convert models and can translate or generate text with few lines of code:

```python
translator = ctranslate2.Translator(translation_model_path)
translator.translate_batch(tokens)

generator = ctranslate2.Generator(generation_model_path)
generator.generate_batch(start_tokens)
```

See the [documentation](https://opennmt.net/CTranslate2) for more information and examples.

## Benchmarks

We translate the En->De test set *newstest2014* with multiple models:

* [OpenNMT-tf WMT14](https://opennmt.net/Models-tf/#translation): a base Transformer trained with OpenNMT-tf on the WMT14 dataset (4.5M lines)
* [OpenNMT-py WMT14](https://opennmt.net/Models-py/#translation): a base Transformer trained with OpenNMT-py on the WMT14 dataset (4.5M lines)
* [OPUS-MT](https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models/en-de#opus-2020-02-26zip): a base Transformer trained with Marian on all OPUS data available on 2020-02-26 (81.9M lines)

The benchmark reports the number of target tokens generated per second (higher is better). The results are aggregated over multiple runs. See the [benchmark scripts](tools/benchmark) for more details and reproduce these numbers.

**Please note that the results presented below are only valid for the configuration used during this benchmark: absolute and relative performance may change with different settings.**

#### CPU

| | Tokens per second | Max. memory | BLEU |
| --- | --- | --- | --- |
| **OpenNMT-tf WMT14 model** | | | |
| OpenNMT-tf 2.25.0 (with TensorFlow 2.8.0) | 342.4 | 2600MB | 26.93 |
| **OpenNMT-py WMT14 model** | | | |
| OpenNMT-py 2.2.0 (with PyTorch 1.9.1) | 458.8 | 1734MB | 26.77 |
| - int8 | 500.1 | 1612MB | 26.72 |
| CTranslate2 2.13.1 | 1217.8 | 1068MB | 26.77 |
| - int16 | 1530.9 | 989MB | 26.82 |
| - int8 | 1787.3 | 773MB | 26.89 |
| - int8 + vmap | 2179.2 | 789MB | 26.62 |
| **OPUS-MT model** | | | |
| Marian 1.11.0 | 756.7 | 13819MB | 27.93 |
| - int16 | 723.6 | 10393MB | 27.65 |
| - int8 | 857.3 | 8169MB | 27.27 |
| CTranslate2 2.13.1 | 993.5 | 901MB | 27.92 |
| - int16 | 1290.4 | 921MB | 27.51 |
| - int8 | 1486.5 | 748MB | 27.71 |

Executed with 8 threads on a [*c5.metal*](https://aws.amazon.com/ec2/instance-types/c5/) Amazon EC2 instance equipped with an Intel(R) Xeon(R) Platinum 8275CL CPU.

#### GPU

| | Tokens per second | Max. GPU memory | Max. CPU memory | BLEU |
| --- | --- | --- | --- | --- |
| **OpenNMT-tf WMT14 model** | | | | |
| OpenNMT-tf 2.25.0 (with TensorFlow 2.8.0) | 1285.7 | 2666MB | 2364MB | 26.93 |
| **OpenNMT-py WMT14 model** | | | | |
| OpenNMT-py 2.2.0 (with PyTorch 1.9.1) | 1220.9 | 3082MB | 3900MB | 26.77 |
| FasterTransformer 4.0 | 2950.8 | 5868MB | 2436MB | 26.77 |
| - float16 | 6499.3 | 3916MB | 2423MB | 26.83 |
| CTranslate2 2.13.1 | 3747.1 | 1264MB | 676MB | 26.77 |
| - int8 | 5306.4 | 976MB | 561MB | 26.83 |
| - float16 | 5367.8 | 816MB | 607MB | 26.78 |
| - int8 + float16 | 6158.7 | 688MB | 563MB | 26.80 |
| **OPUS-MT model** | | | | |
| Marian 1.11.0 | 2221.5 | 3128MB | 1932MB | 27.92 |
| - float16 | 2832.7 | 2986MB | 1713MB | 27.93 |
| CTranslate2 2.13.1 | 3136.3 | 1200MB | 481MB | 27.92 |
| - int8 | 4634.4 | 1008MB | 519MB | 27.89 |
| - float16 | 4708.7 | 816MB | 560MB | 27.85 |
| - int8 + float16 | 5320.3 | 720MB | 515MB | 27.81 |

Executed with CUDA 11 on a [*g4dn.xlarge*](https://aws.amazon.com/ec2/instance-types/g4/) Amazon EC2 instance equipped with a NVIDIA T4 GPU (driver version: 470.82.01).

## Additional resources

* [Documentation](https://opennmt.net/CTranslate2)
* [Forum](https://forum.opennmt.net)
* [Gitter](https://gitter.im/OpenNMT/CTranslate2)
