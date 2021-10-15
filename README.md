[![CI](https://github.com/OpenNMT/CTranslate2/workflows/CI/badge.svg)](https://github.com/OpenNMT/CTranslate2/actions?query=workflow%3ACI) [![PyPI version](https://badge.fury.io/py/ctranslate2.svg)](https://badge.fury.io/py/ctranslate2) [![Gitter](https://badges.gitter.im/OpenNMT/CTranslate2.svg)](https://gitter.im/OpenNMT/CTranslate2?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

# CTranslate2

CTranslate2 is a fast and full-featured inference engine for Transformer models. It aims to provide comprehensive inference features and be the most efficient and cost-effective solution to deploy standard neural machine translation systems on CPU and GPU. It currently supports Transformer models trained with:

* [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)
* [OpenNMT-tf](https://github.com/OpenNMT/OpenNMT-tf)
* [Fairseq](https://github.com/pytorch/fairseq/)

The project is production-oriented and comes with [backward compatibility guarantees](#what-is-the-state-of-this-project), but it also includes experimental features related to model compression and inference acceleration.

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

* **Fast and efficient execution on CPU and GPU**<br/>The execution [is significantly faster and requires less resources](#benchmarks) than general-purpose deep learning frameworks on supported models and tasks thanks to many advanced optimizations: padding removal, batch reordering, in-place operations, caching mechanism, etc.
* **Quantization and reduced precision**<br/>The model serialization and computation support weights with [reduced precision](docs/quantization.md): 16-bit floating points (FP16), 16-bit integers (INT16), and 8-bit integers (INT8).
* **Multiple CPU architectures support**<br/>The project supports x86-64 and ARM64 processors and integrates multiple backends that are optimized for these platforms: [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html), [oneDNN](https://github.com/oneapi-src/oneDNN), [OpenBLAS](https://www.openblas.net/), [Ruy](https://github.com/google/ruy), and [Apple Accelerate](https://developer.apple.com/documentation/accelerate).
* **Automatic CPU detection and code dispatch**<br/>One binary can include multiple backends (e.g. Intel MKL and oneDNN) and instruction set architectures (e.g. AVX, AVX2) that are automatically selected at runtime based on the CPU information.
* **Parallel and asynchronous translations**<br/>Translations can be run efficiently in parallel and asynchronously using multiple GPUs or CPU cores.
* **Dynamic memory usage**<br/>The memory usage changes dynamically depending on the request size while still meeting performance requirements thanks to caching allocators on both CPU and GPU.
* **Lightweight on disk**<br/>Quantization can make the models 4 times smaller on disk with minimal accuracy loss. A full featured Docker image supporting GPU and CPU requires less than 500MB (with CUDA 10.0).
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
* approximating the generation using a pre-compiled [vocabulary map](#how-do-i-generate-a-vocabulary-mapping-file)
* replacing unknown target tokens by source tokens with the highest attention
* biasing translations towards a given prefix (see section 4.2 in [Arivazhagan et al. 2020](https://arxiv.org/abs/1912.03393))
* scoring existing translations

See the [Decoding](docs/decoding.md) documentation for examples.

## Quickstart

The steps below assume a Linux OS and a Python installation (3.6 or above).

**1\. [Install](#installation) the Python package:**

```bash
pip install --upgrade pip
pip install ctranslate2
```

**2\. [Convert](#converting-models) a Transformer model trained with OpenNMT-py, OpenNMT-tf, or Fairseq:**

*a. OpenNMT-py*

```bash
pip install OpenNMT-py

wget https://s3.amazonaws.com/opennmt-models/transformer-ende-wmt-pyOnmt.tar.gz
tar xf transformer-ende-wmt-pyOnmt.tar.gz

ct2-opennmt-py-converter --model_path averaged-10-epoch.pt --output_dir ende_ctranslate2
```

*b. OpenNMT-tf*

```bash
pip install OpenNMT-tf

wget https://s3.amazonaws.com/opennmt-models/averaged-ende-ckpt500k-v2.tar.gz
tar xf averaged-ende-ckpt500k-v2.tar.gz

ct2-opennmt-tf-converter --model_path averaged-ende-ckpt500k-v2 --output_dir ende_ctranslate2 \
    --src_vocab averaged-ende-ckpt500k-v2/wmtende.vocab \
    --tgt_vocab averaged-ende-ckpt500k-v2/wmtende.vocab \
    --model_type TransformerBase
```

*c. Fairseq*

```bash
pip install fairseq

wget https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2
tar xf wmt16.en-de.joined-dict.transformer.tar.bz2

ct2-fairseq-converter --model_path wmt16.en-de.joined-dict.transformer/model.pt \
    --data_dir wmt16.en-de.joined-dict.transformer \
    --output_dir ende_ctranslate2
```

**3\. [Translate](#translating) tokenized inputs with the Python API:**

```python
import ctranslate2

translator = ctranslate2.Translator("ende_ctranslate2/", device="cpu")

# The OpenNMT-py and OpenNMT-tf models use a SentencePiece tokenization:
translator.translate_batch([["▁H", "ello", "▁world", "!"]])

# The Fairseq model uses a BPE tokenization:
translator.translate_batch([["H@@", "ello", "world@@", "!"]])
```

## Installation

### Python package

Python packages are published on [PyPI](https://pypi.org/project/ctranslate2/) for Linux and macOS:

```bash
pip install ctranslate2
```

To translate on GPU you should install the CUDA 11.x toolkit. The macOS version only supports CPU execution.

**Requirements:**

* OS: Linux, macOS
* Python version: >= 3.6
* pip version: >= 19.3
* (optional) CUDA version: 11.x
* (optional) GPU driver version: >= 450.80.02

### Docker images

The [`opennmt/ctranslate2`](https://hub.docker.com/r/opennmt/ctranslate2) repository contains images with prebuilt libraries and clients:

```bash
docker pull opennmt/ctranslate2:latest-ubuntu20.04-cuda11.2
```

The library is installed in `/opt/ctranslate2` and a Python package is installed on the system.

**Requirements:**

* Docker
* (optional) GPU driver version: >= 450.80.02

### Manual compilation

See [Building](#building).

## Converting models

The core CTranslate2 implementation is framework agnostic. The framework specific logic is moved to a conversion step that serializes trained models into a simple binary format.

The following frameworks and models are currently supported:

|     | OpenNMT-tf | OpenNMT-py | Fairseq |
| --- | :---: | :---: | :---: |
| Transformer ([Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)) | ✓ | ✓ | ✓ |
| + relative position representations ([Shaw et al. 2018](https://arxiv.org/abs/1803.02155)) | ✓ | ✓ | |

*If you are using a model that is not listed above, consider opening an issue to discuss future integration.*

The Python package includes a [conversion API](docs/python.md#model-conversion-api) and conversion scripts:

* `ct2-opennmt-py-converter`
* `ct2-opennmt-tf-converter`
* `ct2-fairseq-converter`

The conversion should be run in the same environment as the selected training framework.

### Integrated model conversion

Models can also be converted directly from the supported training frameworks. See their documentation:

* [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/bin/release_model.py)
* [OpenNMT-tf](https://opennmt.net/OpenNMT-tf/serving.html#ctranslate2)

### Quantization and reduced precision

The converters support reducing the weights precision to save on space and possibly accelerate the model execution. See the [Quantization](docs/quantization.md) documentation.

### Adding converters

Each converter should populate a model specification with trained weights coming from an existing model. The model specification declares the variable names and layout expected by the CTranslate2 core engine.

See the existing converters implementation which could be used as a template.

## Translating

The examples use the English-German OpenNMT model converted in the [Quickstart](#quickstart). This model requires a SentencePiece tokenization.

### With the translation client

```bash
echo "▁H ello ▁world !" | docker run -i --rm -v $PWD:/data \
    opennmt/ctranslate2:latest-ubuntu20.04-cuda11.2 --model /data/ende_ctranslate2 --device cpu
```

To translate on GPU, use `docker run --gpus all` and set the option `--device cuda`.

*See `docker run --rm opennmt/ctranslate2:latest-ubuntu20.04-cuda11.2 --help` for additional options.*

### With the Python API

```python
import ctranslate2
translator = ctranslate2.Translator("ende_ctranslate2/", device="cpu")
translator.translate_batch([["▁H", "ello", "▁world", "!"]])
```

*See the [Python reference](docs/python.md) for more advanced usages.*

### With the C++ API

```cpp
#include <iostream>
#include <ctranslate2/translator_pool.h>

int main() {
  const size_t num_translators = 1;
  const size_t num_threads_per_translator = 4;
  ctranslate2::TranslatorPool translator(num_translators,
                                         num_threads_per_translator,
                                         "ende_ctranslate2/",
                                         ctranslate2::Device::CPU);

  const std::vector<std::vector<std::string>> batch = {{"▁H", "ello", "▁world", "!"}};
  const std::vector<ctranslate2::TranslationResult> results = translator.translate_batch(batch);

  for (const auto& token : results[0].output())
    std::cout << token << ' ';
  std::cout << std::endl;
  return 0;
}
```

*See the [`TranslatorPool`](include/ctranslate2/translator_pool.h) class for more advanced usages such as asynchronous translations.*

## Environment variables

Some environment variables can be configured to customize the execution:

* `CT2_CUDA_ALLOCATOR`: Select the CUDA memory allocator. Possible values are: `cub_caching` (default), `cuda_malloc_async` (requires CUDA >= 11.2).
* `CT2_CUDA_ALLOW_FP16`: Allow using FP16 computation on GPU even if the device does not have efficient FP16 support.
* `CT2_CUDA_CACHING_ALLOCATOR_CONFIG`: Tune the CUDA caching allocator (see [Performance](docs/performance.md)).
* `CT2_FORCE_CPU_ISA`: Force CTranslate2 to select a specific instruction set architecture (ISA). Possible values are: `GENERIC`, `AVX`, `AVX2`. Note: this does not impact backend libraries (such as Intel MKL) which usually have their own environment variables to configure ISA dispatching.
* `CT2_TRANSLATORS_CORE_OFFSET`: If set to a non negative value, parallel translators are pinned to cores in the range `[offset, offset + inter_threads]`. Requires compilation with `-DOPENMP_RUNTIME=NONE`.
* `CT2_USE_EXPERIMENTAL_PACKED_GEMM`: Enable the packed GEMM API for Intel MKL (see [Performance](docs/performance.md)).
* `CT2_USE_MKL`: Force CTranslate2 to use (or not) Intel MKL. By default, the runtime automatically decides whether to use Intel MKL or not based on the CPU vendor.
* `CT2_VERBOSE`: Configure the logs verbosity:
  * -3 = off
  * -2 = critical
  * -1 = error
  * 0 = warning (default)
  * 1 = info
  * 2 = debug
  * 3 = trace

When using Python, these variables should be set before importing the `ctranslate2` module, e.g.:

```python
import os
os.environ["CT2_VERBOSE"] = "1"

import ctranslate2
```

## Building

### Docker images

The Docker images build the C++ shared libraries, the translation client, and the Python package. The `docker build` command should be run from the project root directory, e.g.:

```bash
docker build -t opennmt/ctranslate2:latest-ubuntu20.04-cuda11.2 -f docker/Dockerfile .
```

See the `docker/` directory for available images.

### Build options

The project uses [CMake](https://cmake.org/) for compilation. The following options can be set with `-DOPTION=VALUE`:

| CMake option | Accepted values (default in bold) | Description |
| --- | --- | --- |
| BUILD_CLI | OFF, **ON** | Compiles the translation clients |
| BUILD_TESTS | **OFF**, ON | Compiles the tests |
| CMAKE_CXX_FLAGS | *compiler flags* | Defines additional compiler flags |
| CUDA_DYNAMIC_LOADING | **OFF**, ON | Enables the dynamic loading of CUDA libraries at runtime instead of linking against them. Requires Linux and CUDA >= 11. |
| ENABLE_CPU_DISPATCH | OFF, **ON** | Compiles CPU kernels for multiple ISA and dispatches at runtime (should be disabled when explicitly targeting an architecture with the `-march` compilation flag) |
| ENABLE_PROFILING | **OFF**, ON | Enables the integrated profiler (usually disabled in production builds) |
| OPENMP_RUNTIME | **INTEL**, COMP, NONE | Selects or disables the OpenMP runtime (INTEL: Intel OpenMP; COMP: OpenMP runtime provided by the compiler; NONE: no OpenMP runtime) |
| WITH_CUDA | **OFF**, ON | Compiles with the CUDA backend |
| WITH_DNNL | **OFF**, ON | Compiles with the oneDNN backend (a.k.a. DNNL) |
| WITH_MKL | OFF, **ON** | Compiles with the Intel MKL backend |
| WITH_ACCELERATE | **OFF**, ON | Compiles with the Apple Accelerate backend |
| WITH_OPENBLAS | **OFF**, ON | Compiles with the OpenBLAS backend |
| WITH_RUY | **OFF**, ON | Compiles with the Ruy backend |

Some build options require external dependencies:

* `-DWITH_MKL=ON` requires:
  * [Intel MKL](https://software.intel.com/en-us/mkl) (>=2019.5)
* `-DWITH_DNNL=ON` requires:
  * [oneDNN](https://github.com/oneapi-src/oneDNN) (>=1.5)
* `-DWITH_ACCELERATE=ON` requires:
  * [Accelerate](https://developer.apple.com/documentation/accelerate) (only available on macOS)
* `-DWITH_OPENBLAS=ON` requires:
  * [OpenBLAS](https://github.com/xianyi/OpenBLAS)
* `-DWITH_CUDA=ON` requires:
  * [cuBLAS](https://developer.nvidia.com/cublas) (>=10.0)

Multiple backends can be enabled for a single build. When building with both Intel MKL and oneDNN, the backend will be selected at runtime based on the CPU information.

### Example (Ubuntu)

#### Install Intel MKL (optional for GPU only builds)

Use the following instructions to install Intel MKL:

```bash
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo sh -c 'echo "deb https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list'
sudo apt-get update
sudo apt-get install intel-oneapi-mkl-devel
```

See the [Intel MKL documentation](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html) for other installation methods.

#### Install CUDA (optional for CPU only builds)

See the [NVIDIA documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) for information on how to download and install CUDA.

#### Compile

Under the project root, run the following commands:

```bash
git submodule update --init --recursive
mkdir build && cd build
cmake -DWITH_MKL=ON -DWITH_CUDA=ON ..
make -j4
```

(If you did not install one of Intel MKL or CUDA, set its corresponding flag to `OFF` in the CMake command line.)

These steps should produce the `cli/translate` binary. You can try it with the model converted in the [Quickstart](#quickstart) section:

```bash
$ echo "▁H ello ▁world !" | ./cli/translate --model ende_ctranslate2/ --device auto
▁Hallo ▁Welt !
```

## Testing

### C++

To enable the tests, you should configure the project with `cmake -DBUILD_TESTS=ON`. The binary `tests/ctranslate2_test` runs all tests using Google Test. It expects the path to the test data as argument:

```bash
./tests/ctranslate2_test ../tests/data
```

### Python

```bash
# Install the CTranslate2 library.
cd build && make install && cd ..

# Build and install the Python wheel.
cd python
pip install -r install_requirements.txt
python setup.py bdist_wheel
pip install dist/*.whl

# Run the tests with pytest.
pip install -r tests/requirements.txt
pytest tests/test.py
```

Depending on your build configuration, you might need to set `LD_LIBRARY_PATH` if missing libraries are reported when running `tests/test.py`.

## Benchmarks

We compare CTranslate2 with OpenNMT-py and OpenNMT-tf on their pretrained English-German Transformer models (available on the [website](https://opennmt.net/)). **For this benchmark, CTranslate2 models are using the weights of the OpenNMT-py model.**

### Model size

| | Model size |
| --- | --- |
| OpenNMT-py | 542MB |
| OpenNMT-tf | 367MB |
| CTranslate2 | 364MB |
| - int16 | 187MB |
| - float16 | 182MB |
| - int8 | 100MB |
| - int8 + float16 | 95MB |

CTranslate2 models are generally lighter and can go as low as 100MB when quantized to int8. This also results in a fast loading time and noticeable lower memory usage during runtime.

### Results

We translate the test set *newstest2014* and report:

* the number of target tokens generated per second (higher is better)
* the maximum memory usage (lower is better)
* the BLEU score of the detokenized output (higher is better)

See the directory [`tools/benchmark`](tools/benchmark) for more details about the benchmark procedure and how to run it. Also see the [Performance](docs/performance.md) document to further improve CTranslate2 performance.

**Please note that the results presented below are only valid for the configuration used during this benchmark: absolute and relative performance may change with different settings.**

#### CPU

| | Tokens per second | Max. memory | BLEU |
| --- | --- | --- | --- |
| OpenNMT-tf 2.19.0 (with TensorFlow 2.5.0) | 364.1 | 2620MB | 26.93 |
| OpenNMT-py 2.1.2 (with PyTorch 1.9.0) | 472.6 | 1856MB | 26.77 |
| - int8 | 510.4 | 1712MB | 26.80 |
| CTranslate2 2.3.0 | 1182.3 | 1037MB | 26.77 |
| - int16 | 1532.0 | 954MB | 26.83 |
| - int8 | 1785.2 | 810MB | 26.86 |
| - int8 + vmap | 2263.4 | 692MB | 26.70 |

Executed with 8 threads on a [*c5.metal*](https://aws.amazon.com/ec2/instance-types/c5/) Amazon EC2 instance equipped with an Intel(R) Xeon(R) Platinum 8275CL CPU.

#### GPU

| | Tokens per second | Max. GPU memory | Max. CPU memory | BLEU |
| --- | --- | --- | --- | --- |
| OpenNMT-tf 2.19.0 (with TensorFlow 2.5.0) | 1815.2 | 2660MB | 1724MB | 26.93 |
| OpenNMT-py 2.1.2 (with PyTorch 1.9.0) | 1536.7 | 3046MB | 2987MB | 26.77 |
| CTranslate2 2.3.0 | 3696.7 | 1234MB | 555MB | 26.77 |
| - int8 | 5201.9 | 946MB | 565MB | 26.82 |
| - float16 | 5303.5 | 818MB | 607MB | 26.75 |
| - int8 + float16 | 5824.3 | 722MB | 566MB | 26.88 |

Executed with CUDA 11 on a [*g4dn.xlarge*](https://aws.amazon.com/ec2/instance-types/g4/) Amazon EC2 instance equipped with a NVIDIA T4 GPU (driver version: 460.73.01).

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

### How does it relate to the original CTranslate project?

The original [CTranslate](https://github.com/OpenNMT/CTranslate) project shares a similar goal which is to provide a custom execution engine for OpenNMT models that is lightweight and fast. However, it has some limitations that were hard to overcome:

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
  * `ctranslate2.converters.FairseqConverter`
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

**CPU**

CTranslate2 supports x86-64 and ARM64 processors. It includes optimizations for AVX, AVX2, and NEON and supports multiple BLAS backends that should be selected based on the target platform (see [Building](#building)).

Prebuilt binaries are designed to run on any x86-64 processors supporting at least SSE 4.2. The binaries implement runtime dispatch to select the best backend and instruction set architecture (ISA) for the platform. In particular, they are compiled with both [Intel MKL](https://software.intel.com/en-us/mkl) and [oneDNN](https://github.com/oneapi-src/oneDNN) so that Intel MKL is only used on Intel processors where it performs best, whereas oneDNN is used on other x86-64 processors such as AMD.

**GPU**

CTranslate2 supports NVIDIA GPUs with a Compute Capability greater or equal to 3.5.

The driver requirement depends on the CUDA version. See the [CUDA Compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/index.html) for more information.

### What are the known limitations?

The current approach only exports the weights from existing models and redefines the computation graph via the code. This implies a strong assumption of the graph architecture executed by the original framework.

We could ease this assumption by supporting ONNX as model parts.

### What are the future plans?

There are many ways to make this project better and even faster. See the open issues for an overview of current and planned features. Here are some things we would like to get to:

* Support of running ONNX graphs

### What is the difference between `intra_threads` and `inter_threads`?

* `intra_threads` is the number of OpenMP threads that is used per translation: increase this value to decrease the latency.
* `inter_threads` is the maximum number of CPU translations executed in parallel: increase this value to increase the throughput. Even though the model data are shared, this execution mode will increase the memory usage as some internal buffers are duplicated for thread safety.

The total number of computing threads launched by the process is `inter_threads * intra_threads`.

Note that these options are only defined for CPU translation and are forced to 1 when executing on GPU. Parallel translations on GPU require multiple GPUs. See the option `device_index` that accepts multiple device IDs.

### Do you provide a translation server?

The [OpenNMT-py REST server](https://forum.opennmt.net/t/simple-opennmt-py-rest-server/1392) is able to serve CTranslate2 models. See the [code integration](https://github.com/OpenNMT/OpenNMT-py/commit/91d5d57142b9aa0a0859fbfa0dd94f301f56f879) to learn more.

### How do I generate a vocabulary mapping file?

The vocabulary mapping file (a.k.a. *vmap*) maps source N-grams to a list of target tokens. During translation, the target vocabulary will be dynamically reduced to the union of all target tokens associated with the N-grams from the batch to translate.

It is a text file where each line has the following format:

```text
src_1 src_2 ... src_N<TAB>tgt_1 tgt_2 ... tgt_K
```

If the source N-gram is empty (N = 0), the assiocated target tokens will always be included in the reduced vocabulary.

See [here](https://github.com/OpenNMT/papers/tree/master/WNMT2018/vmap) for an example on how to generate this file. The file can then be passed to the converter script to be included in the model directory (see option `--vocab_mapping`) and can be used during translation after enabling the `use_vmap` translation option.
