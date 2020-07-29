[![Build Status](https://travis-ci.com/OpenNMT/CTranslate2.svg?branch=master)](https://travis-ci.com/OpenNMT/CTranslate2) [![PyPI version](https://badge.fury.io/py/ctranslate2.svg)](https://badge.fury.io/py/ctranslate2) [![Gitter](https://badges.gitter.im/OpenNMT/CTranslate2.svg)](https://gitter.im/OpenNMT/CTranslate2?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

# CTranslate2

CTranslate2 is a fast inference engine for [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) and [OpenNMT-tf](https://github.com/OpenNMT/OpenNMT-tf) models supporting both CPU and GPU execution. The goal is to provide comprehensive inference features and be the most efficient and cost-effective solution to deploy standard neural machine translation systems such as Transformer models.

The project is production-oriented and comes with [backward compatibility guarantees](#what-is-the-state-of-this-project), but has also experimental features related to model compression and inference acceleration.

**Table of contents**

1. [Key features](#key-features)
1. [Quickstart](#quickstart)
1. [Installation](#installation)
1. [Converting models](#converting-models)
1. [Translating](#translating)
1. [Environment variables](#environment-variables)
1. [Building](#building)
1. [Testing](#testing)
1. [Benchmarks](#benchmarks)
1. [Frequently asked questions](#frequently-asked-questions)

## Key features

* **Fast and efficient runtime**<br/>The runtime aims to be faster and lighter than a general-purpose deep learning framework: it is [up to 4x faster](#benchmarks) than OpenNMT-py on standard translation tasks.
* **Quantization and reduced precision**<br/>The model serialization and computation support weights with reduced precision: 16-bit floating points (FP16), 16-bit integers, and 8-bit integers.
* **Parallel translations**<br/>CPU translations can be run efficiently in parallel without duplicating the model data in memory.
* **Dynamic memory usage**<br/>The memory usage changes dynamically depending on the request size while still meeting performance requirements thanks to caching allocators on both CPU and GPU.
* **Automatic CPU detection and code dispatch**<br/>The fastest code path is selected at runtime based on the CPU (Intel or AMD) and the supported instruction set architectures (AVX, AVX2, or AVX512).
* **Ligthweight on disk**<br/>Models can be quantized below 100MB with minimal accuracy loss. A full featured Docker image supporting GPU and CPU requires less than 1GB.
* **Simple integration**<br/>The project has few dependencies and exposes [translation APIs](#translating) in Python and C++ to cover most integration needs.
* **Interactive decoding**<br/>[Advanced decoding features](docs/decoding.md) allow autocompleting a partial translation and returning alternatives at a specific location in the translation.

Some of these features are difficult to achieve with standard deep learning frameworks and are the motivation for this project.

### Supported decoding options

The translation API supports several decoding options:

* decoding with greedy or beam search
* random sampling from the output distribution
* translating with a known target prefix
* returning alternatives at a specific location in the target
* constraining the decoding length
* returning multiple translation hypotheses
* returning attention vectors
* approximating the generation using a pre-compiled [vocabulary map](#how-can-i-generate-a-vocabulary-mapping-file)

See the [Decoding](docs/decoding.md) documentation for examples.

## Quickstart

The steps below assume a Linux OS and a Python installation.

1\. **[Install](#installation) the Python package**:

```bash
pip install --upgrade pip
pip install ctranslate2
```

2\. **[Convert](#converting-models) a model trained with OpenNMT-py or OpenNMT-tf**, for example the pretrained Transformer model (choose one of the two models):

*a. OpenNMT-py*

```bash
pip install OpenNMT-py

wget https://s3.amazonaws.com/opennmt-models/transformer-ende-wmt-pyOnmt.tar.gz
tar xf transformer-ende-wmt-pyOnmt.tar.gz

ct2-opennmt-py-converter --model_path averaged-10-epoch.pt --model_spec TransformerBase \
    --output_dir ende_ctranslate2
```

*b. OpenNMT-tf*

```bash
pip install OpenNMT-tf

wget https://s3.amazonaws.com/opennmt-models/averaged-ende-export500k-v2.tar.gz
tar xf averaged-ende-export500k-v2.tar.gz

ct2-opennmt-tf-converter --model_path averaged-ende-export500k-v2 --model_spec TransformerBase \
    --output_dir ende_ctranslate2
```

3\. **[Translate](#translating) tokenized inputs**, for example with the Python API:

```python
>>> import ctranslate2
>>> translator = ctranslate2.Translator("ende_ctranslate2/")
>>> translator.translate_batch([["▁H", "ello", "▁world", "!"]])
```

## Installation

### Python package

The [`ctranslate2`](https://pypi.org/project/ctranslate2/) Python package will get you started in converting and executing models:

```bash
pip install ctranslate2
```

The package published on PyPI only supports CPU execution at the moment. Consider using a Docker image for GPU support (see below).

**Requirements:**

* OS: Linux
* pip version: >= 19.0

### Docker images

The [`opennmt/ctranslate2`](https://hub.docker.com/r/opennmt/ctranslate2) repository contains images for multiple Linux distributions, with or without GPU support:

```bash
docker pull opennmt/ctranslate2:latest-ubuntu18-cuda10.2
```

The images include:

* a translation client to directly translate files
* Python 3 packages (with GPU support)
* `libctranslate2.so` library development files

### Manual compilation

See [Building](#building).

## Converting models

The core CTranslate2 implementation is framework agnostic. The framework specific logic is moved to a conversion step that serializes trained models into a simple binary format.

The following frameworks and models are currently supported:

|     | OpenNMT-tf | OpenNMT-py |
| --- | :---: | :---: |
| Transformer ([Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)) | ✓ | ✓ |
| + relative position representations ([Shaw et al. 2018](https://arxiv.org/abs/1803.02155)) | ✓ | ✓ |

*If you are using a model that is not listed above, consider opening an issue to discuss future integration.*

Conversion scripts are parts of the Python package and should be run in the same environment as the selected training framework:

* `ct2-opennmt-py-converter`
* `ct2-opennmt-tf-converter`

The [converter Python API](docs/python.md#model-conversion-api) can also be used to convert Transformer models with any number of layers, hidden dimensions, and attention heads.

### Integrated model conversion

Models can also be converted directly from the supported training frameworks. See their documentation:

* [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/bin/release_model.py)
* [OpenNMT-tf](https://opennmt.net/OpenNMT-tf/serving.html#ctranslate2)

### Quantization and reduced precision

The converters support reducing the weights precision to save on space and possibly accelerate the model execution. The `--quantization` option accepts the following values:

* `int8`
* `int16`
* `float16`

However, some execution settings are not (yet) optimized for all computation types. The following table documents the actual type used during the computation depending on the model type:

| Model type | GPU (NVIDIA) | CPU (Intel) | CPU (AMD) |
| ---------- | ------------ | ----------- | --------- |
| float16    | float16 (\*) | float       | float     |
| int16      | float16 (\*) | int16       | int8      |
| int8       | int8 (\*\*)  | int8        | int8      |

*(\*) for Compute Capability >= 7.0, (\*\*) for Compute Capability >= 7.0 or == 6.1.*

The computation type can also be configured later, when starting a translation instance. See the `compute_type` argument on translation clients.

**Notes:**

* Integer quantization is only supported for GEMM-based layers and embeddings

### Adding converters

Each converter should populate a model specification with trained weights coming from an existing model. The model specification declares the variable names and layout expected by the CTranslate2 core engine.

See the existing converters implementation which could be used as a template.

## Translating

The examples use the English-German model converted in the [Quickstart](#quickstart). This model requires a SentencePiece tokenization.

### With the translation client

```bash
echo "▁H ello ▁world !" | docker run --gpus=all -i --rm -v $PWD:/data \
    opennmt/ctranslate2:latest-ubuntu18-cuda10.2 --model /data/ende_ctranslate2 --device cuda
```

*See `docker run --rm opennmt/ctranslate2:latest-ubuntu18-cuda10.2 --help` for additional options.*

### With the Python API

```python
>>> import ctranslate2
>>> translator = ctranslate2.Translator("ende_ctranslate2/")
>>> translator.translate_batch([["▁H", "ello", "▁world", "!"]])
```

*See the [Python reference](docs/python.md) for more advanced usages.*

### With the C++ API

```cpp
#include <iostream>
#include <ctranslate2/translator.h>

int main() {
  ctranslate2::Translator translator("ende_ctranslate2/");
  ctranslate2::TranslationResult result = translator.translate({"▁H", "ello", "▁world", "!"});

  for (const auto& token : result.output())
    std::cout << token << ' ';
  std::cout << std::endl;
  return 0;
}
```

*See the [Translator class](include/ctranslate2/translator.h) for more advanced usages, and the [TranslatorPool class](include/ctranslate2/translator_pool.h) for running translations in parallel.*

## Environment variables

Some environment variables can be configured to customize the execution:

* `CT2_CUDA_ALLOW_FLOAT16`: Allow using FP16 computation on GPU even if the device does not have efficient FP16 support.
* `CT2_CUDA_CACHING_ALLOCATOR_CONFIG`: Tune the CUDA caching allocator (see [Performance](docs/performance.md)).
* `CT2_FORCE_CPU_ISA`: Force CTranslate2 to select a specific instruction set architecture (ISA). Possible values are: `GENERIC`, `AVX`, `AVX2`. Note: this does not impact backend libraries (such as Intel MKL) which usually have their own environment variables to configure ISA dispatching.
* `CT2_TRANSLATORS_CORE_OFFSET`: If set to a non negative value, parallel translators are pinned to cores in the range `[offset, offset + inter_threads]`. Requires `intra_threads` to 1.
* `CT2_USE_EXPERIMENTAL_PACKED_GEMM`: Enable the packed GEMM API for Intel MKL (see [Performance](docs/performance.md)).
* `CT2_USE_MKL`: Force CTranslate2 to use (or not) Intel MKL. By default, the runtime automatically decides whether to use Intel MKL or not based on the CPU vendor.
* `CT2_VERBOSE`: Enable some verbose logs to help debugging the run configuration.

## Building

### Dependencies

Backends can be enabled or disabled during the CMake configuration. CTranslate2 supports multiple backends in a single binary:

* `-DWITH_MKL=ON` requires:
  * [Intel MKL](https://software.intel.com/en-us/mkl) (>=2019.5)
* `-DWITH_DNNL=ON` requires:
  * [oneDNN](https://github.com/oneapi-src/oneDNN) (>=1.5)
* `-DWITH_CUDA=ON` requires:
  * [TensorRT](https://developer.nvidia.com/tensorrt) (>=6.0,<7.0)
  * [cuBLAS](https://developer.nvidia.com/cublas) (>=10.0)
  * [cuDNN](https://developer.nvidia.com/cudnn) (>=7.5)

When building with both Intel MKL and oneDNN, the backend will be selected at runtime based on the CPU information.

### Docker images

The Docker images are self contained and build the code from the active directory. The `build` command should be run from the project root directory, e.g.:

```bash
docker build -t opennmt/ctranslate2:latest-ubuntu18 -f docker/Dockerfile.ubuntu .
```

When building GPU images, the CUDA version can be selected with `--build-arg CUDA_VERSION=10.2`.

See the `docker/` directory for available images.

### Binaries (Ubuntu)

This minimal installation only enables CPU execution with Intel MKL. For more advanced usages and GPU support, see how the [Ubuntu GPU Dockerfile](docker/Dockerfile.ubuntu-gpu) is defined.

#### Install Intel MKL

Use the following instructions to install Intel MKL:

```bash
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
sudo sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
sudo apt-get update
sudo apt-get install intel-mkl-64bit-2020.2-108
```

Go to https://software.intel.com/en-us/articles/installing-intel-free-libs-and-python-apt-repo for more details.

#### Compile

Under the project root, run the following commands:

```bash
mkdir build && cd build
cmake ..
make -j4
```

The `cli/translate` binary will be generated. You can try it with the model converted in the [Quickstart](#quickstart) section:

```bash
echo "▁H ello ▁world !" | ./cli/translate --model ende_ctranslate2/
```

The result `▁Hallo ▁Welt !` should be displayed.

## Testing

### C++

To enable the tests, you should configure the project with `cmake -DWITH_TESTS=ON`. The binary `tests/ctranslate2_test` runs all tests using Google Test. It expects the path to the test data as argument:

```bash
./tests/ctranslate2_test ../tests/data
```

## Benchmarks

We compare CTranslate2 with OpenNMT-py and OpenNMT-tf on their pretrained English-German Transformer models (available on the [website](https://opennmt.net/)). **For this benchmark, CTranslate2 models are using the weights of the OpenNMT-py model.**

### Model size

| | Model size |
| --- | --- |
| CTranslate2 (int8) | 110MB |
| CTranslate2 (int16) | 197MB |
| OpenNMT-tf | 367MB |
| CTranslate2 (float) | 374MB |
| OpenNMT-py | 542MB |

CTranslate2 models are generally lighter and can go as low as 100MB when quantized to int8. This also results in a fast loading time and noticeable lower memory usage during runtime.

### Speed

We translate the test set *newstest2014* and report:

* the number of target tokens generated per second (higher is better)
* the BLEU score of the detokenized output (higher is better)

Translations are running beam search with a size of 4 and a maximum batch size of 32. CPU translations are using 4 threads.

**Please note that the results presented below are only valid for the configuration used during this benchmark: absolute and relative performance may change with different settings.**

| | CPU (i7-7700) | GPU (GTX 1080) | GPU (GTX 1080 Ti) | GPU (RTX 2080 Ti) | BLEU |
| --- | --- | --- | --- | --- | --- |
| OpenNMT-py 1.1.1 | 179.1 | 1510.0 | 1709.3 | 1406.2 | 26.69 |
| OpenNMT-tf 2.9.1 | 217.6 | 1659.2 | 1762.8 | 1628.3 | 26.90 |
| CTranslate2 1.10.0 | 389.4 | 3081.3 | 3388.0 | 4196.2 | 26.69 |
| - int16 | 413.6 | | | | 26.68 |
| - int16 + vmap | 527.6 | | | | 26.63 |
| - int8 | 508.3 | 2654.8 | 2734.6 | 3143.4 | 26.84 |
| - int8 + vmap | 646.2 | 2921.5 | 2992.1 | 3319.9 | 26.59 |

#### Comments

* On GPU, int8 quantization is generally slower as the runtime overhead of int8<->float conversions is presently too high compared to the actual computation.
* On CPU, performance gains of quantized runs can be greater depending on settings such as the number of threads, batch size, beam size, etc.
* In addition to possible performance gains, quantization results in a much lower memory usage and can also act as a regularizer (hence the higher BLEU score in some cases).

### Memory usage

We don't have numbers comparing memory usage yet. However, past experiments showed that CTranslate2 usually requires up to 2x less memory than OpenNMT-py.

## Frequently asked questions

* [How does it relate to the original CTranslate project?](#how-does-it-relate-to-the-original-ctranslate-project)
* [What is the state of this project?](#what-is-the-state-of-this-project)
* [Why and when should I use this implementation instead of PyTorch or TensorFlow?](#why-and-when-should-i-use-this-implementation-instead-of-pytorch-or-tensorflow)
* [What hardware is supported?](#what-hardware-is-supported)
* [What are the known limitations?](#what-are-the-known-limitations)
* [What are the future plans?](#what-are-the-future-plans)
* [What is the difference between `intra_threads` and `inter_threads`?](#what-is-the-difference-between-intra_threads-and-inter_threads)
* [Do you provide a translation server?](#do-you-provide-a-translation-server)
* [How do I generate a vocabulary mapping file?](#how-do-i-generate-a-vocabulary-mapping-file)

### How does it relate to the original [CTranslate](https://github.com/OpenNMT/CTranslate) project?

The original CTranslate project shares a similar goal which is to provide a custom execution engine for OpenNMT models that is lightweight and fast. However, it has some limitations that were hard to overcome:

* a strong dependency on LuaTorch and OpenNMT-lua, which are now both deprecated in favor of other toolkits;
* a direct reliance on [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page), which introduces heavy templating and a limited GPU support.

CTranslate2 addresses these issues in several ways:

* the core implementation is framework agnostic, moving the framework specific logic to a model conversion step;
* the internal operators follow the ONNX specifications as much as possible for better future-proofing;
* the call to external libraries (Intel MKL, cuBLAS, etc.) occurs as late as possible in the execution to not rely on a library specific logic.

### What is the state of this project?

The implementation has been generously tested in [production environment](https://translate.systran.net/) so people can rely on it in their application. The project versioning follows [Semantic Versioning 2.0.0](https://semver.org/). The following APIs are covered by backward compatibility guarantees:

* Converted models
* Python converters options
* Python symbols:
  * `ctranslate2.Translator`
  * `ctranslate2.converters.OpenNMTPyConverter`
  * `ctranslate2.converters.OpenNMTTFConverter`
* C++ symbols:
  * `ctranslate2::models::Model`
  * `ctranslate2::TranslationOptions`
  * `ctranslate2::TranslationResult`
  * `ctranslate2::Translator`
  * `ctranslate2::TranslatorPool`
* C++ translation client options

Other APIs are expected to evolve to increase efficiency, genericity, and model support.

### Why and when should I use this implementation instead of PyTorch or TensorFlow?

Here are some scenarios where this project could be used:

* You want to accelarate standard translation models for production usage, especially on CPUs.
* You need to embed translation models in an existing C++ application without adding large dependencies.
* Your application requires custom threading and memory usage control.
* You want to reduce the model size on disk and/or memory.

However, you should probably **not** use this project when:

* You want to train custom architectures not covered by this project.
* You see no value in the key features listed at the top of this document.

### What hardware is supported?

The supported hardware mostly depends on the external libraries used for acceleration.

**CPU**

We recommend using a recent Intel CPU and [Intel MKL](https://software.intel.com/en-us/mkl) for maximum performance.

However, Intel MKL is known to run poorly on AMD CPUs. To improve AMD support, we recommend enabling the [oneDNN](https://github.com/oneapi-src/oneDNN) backend that will be automatically selected at runtime. oneDNN is included in all pre-built binaries of CTranslate2.

Optimized execution on ARM is a future work (contributions are welcome!).

**GPU**

CTranslate2 currently requires a NVIDIA GPU with a Compute Capability greater or equal to 3.0 (Kepler). FP16 execution requires a Compute Capability greater or equal to 7.0.

The driver requirement depends on the CUDA version, see the [CUDA Compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/index.html) for more information.

### What are the known limitations?

The current approach only exports the weights from existing models and redefines the computation graph via the code. This implies a strong assumption of the graph architecture executed by the original framework.

We are actively looking to ease this assumption by supporting ONNX as model parts.

### What are the future plans?

There are many ways to make this project better and even faster. See the open issues for an overview of current and planned features. Here are some things we would like to get to:

* Increased support of INT8 quantization, for example by quantizing more layers
* Support of running ONNX graphs
* Optimizations for ARM CPUs
* Support GPU execution with the Python packages published on PyPI

### What is the difference between `intra_threads` and `inter_threads`?

* `intra_threads` is the number of OpenMP threads that is used per translation: increase this value to decrease the latency.
* `inter_threads` is the maximum number of translations executed in parallel: increase this value to increase the throughput (this will also increase the memory usage as some internal buffers are duplicated for thread safety).

The total number of computing threads launched by the process is summarized by this formula:

```text
num_threads = inter_threads * intra_threads
```

Note that these options are only defined for CPU translation and are forced to 1 when executing on GPU.

### Do you provide a translation server?

The [OpenNMT-py REST server](https://forum.opennmt.net/t/simple-opennmt-py-rest-server/1392) is able to serve CTranslate2 models. See the [code integration](https://github.com/OpenNMT/OpenNMT-py/commit/91d5d57142b9aa0a0859fbfa0dd94f301f56f879) to learn more.

### How do I generate a vocabulary mapping file?

See [here](https://github.com/OpenNMT/papers/tree/master/WNMT2018/vmap).
