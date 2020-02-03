[![Build Status](https://travis-ci.com/OpenNMT/CTranslate2.svg?branch=master)](https://travis-ci.com/OpenNMT/CTranslate2) [![Gitter](https://badges.gitter.im/OpenNMT/CTranslate2.svg)](https://gitter.im/OpenNMT/CTranslate2?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

# CTranslate2

CTranslate2 is an optimized inference engine for [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) and [OpenNMT-tf](https://github.com/OpenNMT/OpenNMT-tf) models supporting both CPU and GPU execution. This project is geared towards efficient serving of standard translation models but is also a place for experimentation around model compression and inference acceleration.

**Table of contents**

1. [Key features](#key-features)
1. [Quickstart](#quickstart)
1. [Installation](#installation)
1. [Converting models](#converting-models)
1. [Translating](#translating)
1. [Building](#building)
1. [Testing](#testing)
1. [Benchmarks](#benchmarks)
1. [Frequently asked questions](#frequently-asked-questions)

## Key features

* **Fast execution**<br/>The execution aims to be faster than a general purpose deep learning framework: on standard translation tasks, it is [up to 4x faster](#benchmarks) than OpenNMT-py.
* **Model quantization**<br/>Support INT16 quantization on CPU and INT8 quantization (experimental) on CPU and GPU.
* **Parallel translation**<br/>Translations can be run efficiently in parallel without duplicating the model data in memory.
* **Dynamic memory usage**<br/>The memory usage changes dynamically depending on the request size while still meeting performance requirements thanks to caching allocators on both CPU and GPU.
* **Automatic instruction set dispatch**<br/>When using Intel MKL, the dispatch to the optimal instruction set is done at runtime.
* **Ligthweight on disk**<br/>Models can be quantized below 100MB with minimal accuracy loss. A full featured Docker image supporting GPU and CPU requires less than 1GB.
* **Easy to use translation APIs**<br/>The project exposes [translation APIs](#translating) in Python and C++ to cover most integration needs.

Some of these features are difficult to achieve with standard deep learning frameworks and are the motivation for this project.

### Supported decoding options

The translation API supports several decoding options:

* decoding with greedy or beam search
* random sampling from the output distribution
* translating with a known target prefix
* constraining the decoding length
* returning multiple translation hypotheses
* returning attention vectors
* approximating the generation using a pre-compiled [vocabulary map](#how-can-i-generate-a-vocabulary-mapping-file)

## Quickstart

1\. **[Install](#installation) the Python package**:

```bash
pip install ctranslate2
```

2\. **[Convert](#converting-models) a model trained with OpenNMT-py or OpenNMT-tf**, for example the pretrained Transformer model:

*OpenNMT-py*

```bash
wget https://s3.amazonaws.com/opennmt-models/transformer-ende-wmt-pyOnmt.tar.gz
tar xf transformer-ende-wmt-pyOnmt.tar.gz

ct2-opennmt-py-converter --model_path averaged-10-epoch.pt --model_spec TransformerBase \
    --output_dir ende_ctranslate2
```

*OpenNMT-tf*

```bash
wget https://s3.amazonaws.com/opennmt-models/averaged-ende-export500k-v2.tar.gz
tar xf averaged-ende-export500k-v2.tar.gz

ct2-opennmt-tf-converter --model_path averaged-ende-export500k-v2 --model_spec TransformerBase \
    --output_dir ende_ctranslate2
```

3\. **[Translate](#translating) tokenized inputs**, for example with the Python API:

```python
import ctranslate2
translator = ctranslate2.Translator("ende_ctranslate2/")
outputs = translator.translate_batch([["▁H", "ello", "▁world", "!"]])
print(outputs[0][0])
```

## Installation

### Python package

The [`ctranslate2`](https://pypi.org/project/ctranslate2/) Python package will get you started in converting and executing models (CPU only):

```bash
pip install ctranslate2
```

### Docker images

The [`opennmt/ctranslate2`](https://hub.docker.com/r/opennmt/ctranslate2) repository contains images for multiple Linux distributions, with or without GPU support:

```bash
docker pull opennmt/ctranslate2:latest-ubuntu18-gpu
```

The images include:

* a translation client to directly translate files (default entrypoint)
* Python 2 and 3 packages (with GPU support)
* `libctranslate2.so` library development files

### Manual compilation

See [Building](#building).

## Converting models

The core CTranslate2 implementation is framework agnostic. The framework specific logic is moved to a conversion step that serializes trained models into a simple binary format.

The following frameworks and models are currently supported:

|     | [OpenNMT-tf](python/ctranslate2/converters/opennmt_tf.py) | [OpenNMT-py](python/ctranslate2/converters/opennmt_py.py) |
| --- | :---: | :---: |
| Transformer | ✓ | ✓ |

*If you are using a model that is not listed above, consider opening an issue to discuss future integration.*

Conversion scripts are parts of the Python package and should be run in the same environment as the target training framework:

* `ct2-opennmt-py-converter`
* `ct2-opennmt-tf-converter`

The [converter Python API](docs/python.md#model-conversion-api) can also be used to convert Transformer models with any number of layers, hidden dimensions, and attention heads.

### Quantization

The converters support model quantization which is a way to reduce the model size and accelerate its execution. The `--quantization` option accepts the following values:

* `int8`
* `int16`

However, some execution settings are not (yet) optimized for all quantization types. The following table documents the actual types used during the computation:

| Model type | GPU   | CPU (with MKL) |
| ---------- | ----- | -------------- |
| int16      | float | int16          |
| int8       | int8  | int8           |

Quantization can also be configured later when starting a translation instance. See the `compute_type` argument on translation clients.

**Notes:**

* only GEMM-based layers and embeddings are currently quantized

### Adding converters

Each converter should populate a model specification with trained weights coming from an existing model. The model specification declares the variable names and layout expected by the CTranslate2 core engine.

See the existing converters implementation which could be used as a template.

## Translating

The examples use the English-German model converted in the [Quickstart](#quickstart). This model requires a SentencePiece tokenization.

### With the translation client

```bash
echo "▁H ello ▁world !" | nvidia-docker run -i --rm -v $PWD:/data \
    opennmt/ctranslate2:latest-ubuntu18-gpu --model /data/ende_ctranslate2 --device cuda
```

*See `docker run --rm opennmt/ctranslate2:latest-ubuntu18-gpu --help` for additional options.*

### With the Python API

```python
>>> import ctranslate2
>>> translator = ctranslate2.Translator("ende_ctranslate2/")
>>> translator.translate_batch([["▁H", "ello", "▁world", "!"]])
```

*See the [Python reference](docs/python.md) for more advanced usage.*

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

*See the [Translator class](include/ctranslate2/translator.h) for more advanced usage, and the [TranslatorPool class](include/ctranslate2/translator_pool.h) for running translations in parallel.*

## Building

### Dependencies

CTranslate2 uses the following external libraries for acceleration:

* CPU requires:
  * [Intel MKL](https://software.intel.com/en-us/mkl) (>=2019.5)
* GPU requires:
  * [CUB](https://nvlabs.github.io/cub/) (>=1.8)
  * [TensorRT](https://developer.nvidia.com/tensorrt) (>=6.0,<7.0)
  * [Thrust](https://docs.nvidia.com/cuda/thrust/index.html) (==1.9.3)
  * [cuBLAS](https://developer.nvidia.com/cublas) (>=10.0)
  * [cuDNN](https://developer.nvidia.com/cudnn) (>=7.5)

CTranslate2 supports compiling for CPU only, GPU only, or both.

### Docker images

The Docker images are self contained and build the code from the active directory. The `build` command should be run from the project root directory, e.g.:

```bash
docker build -t opennmt/ctranslate2:latest-ubuntu18 -f docker/Dockerfile.ubuntu .
```

See the `docker/` directory for available images.

### Binaries (Ubuntu)

Intel MKL is the minimum requirement for building CTranslate2. The instructions below assume an Ubuntu system.

**Note:** This minimal installation only enables CPU execution. For GPU support, see how the [GPU Dockerfile](docker/Dockerfile.ubuntu-gpu) is defined.

#### Install Intel MKL

Use the following instructions to install Intel MKL:

```bash
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
sudo sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
sudo apt-get update
sudo apt-get install intel-mkl-64bit-2020.0-088
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

Google Test is used to run the C++ tests:

1. Download the [latest release](https://github.com/google/googletest/releases/tag/release-1.8.1)
2. Unzip into an empty folder
3. Go under the decompressed folder (where `CMakeLists.txt` is located)
4. Run the following commands:

```bash
cmake .
sudo make install
sudo ln -s  /usr/local/lib/libgtest.a /usr/lib/libgtest.a
sudo ln -s  /usr/local/lib/libgtest_main.a /usr/lib/libgtest_main.a
```

Then configure the project with `cmake -DWITH_TESTS=ON` to produce the test executable `tests/ctranslate2_test`. The binary expects the path to the test data as argument:

```bash
./tests/ctranslate2_test ../tests/data
```

## Benchmarks

We compare CTranslate2 with OpenNMT-py and OpenNMT-tf on their pretrained English-German Transformer models (available on the [website](http://opennmt.net/)). **For this benchmark, CTranslate2 models are using the weights of the OpenNMT-py model.**

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

Unless otherwise noted, translations are running beam search with a size of 4 and a maximum batch size of 32.

**Please note that the results presented below are only valid for the configuration used during this benchmark: absolute and relative performance may change with different settings.**

#### GPU

Configuration:

* **GPU:** NVIDIA Tesla V100
* **CUDA:** 10.0
* **Driver:** 410.48

| | Tokens/s | BLEU |
| --- | --- | --- |
| CTranslate2 1.2.1 | 3917.32 | 26.70 |
| CTranslate2 1.2.1 (int8) | 2519.24 | 26.80 |
| OpenNMT-tf 1.25.0 | 1338.26 | 26.90 |
| OpenNMT-py 0.9.2 | 980.44 | 26.69 |

#### CPU

Configuration:

* **CPU:** Intel(R) Core(TM) i7-6700K CPU @ 4.00GHz (with AVX2)
* **Number of threads:** 4 "intra threads", 1 "inter threads" ([what's the difference?](#what-is-the-difference-between-intra_threads-and-inter_threads))

| | Tokens/s | BLEU |
| --- | --- | --- |
| CTranslate2 1.2.1 (int8 + vmap) | 591.78 | 26.59 |
| CTranslate2 1.2.1 (int16 + vmap) | 514.36 | 26.63 |
| CTranslate2 1.2.1 (int8) | 448.21 | 26.84 |
| CTranslate2 1.2.1 (int16) | 395.46 | 26.68 |
| CTranslate2 1.2.1 (float) | 381.31 | 26.69 |
| OpenNMT-py 0.9.2 | 241.92 | 26.69 |
| OpenNMT-tf 1.25.0 | 119.34 | 26.90 |

#### Comments

* Both CTranslate2 and OpenNMT-py drop finished translations from the batch which is especially benefitial on CPU.
* On GPU, int8 quantization is generally slower as the runtime overhead of int8<->float conversions is presently too high compared to the actual computation.
* On CPU, performance gains of quantized runs can be greater depending on settings such as the number of threads, batch size, beam size, etc.
* In addition to possible performance gains, quantization results in a much lower memory usage and can also act as a regularizer (hence the higher BLEU score in some cases).

### Memory usage

We don't have numbers comparing memory usage yet. However, past experiments showed that CTranslate2 usually requires up to 2x less memory than OpenNMT-py.

## Frequently asked questions

### How does it relate to the original [CTranslate](https://github.com/OpenNMT/CTranslate) project?

The original CTranslate project shares a similar goal which is to provide a custom execution engine for OpenNMT models that is lightweight and fast. However, it has some limitations that were hard to overcome:

* a strong dependency on LuaTorch and OpenNMT-lua, which are now both deprecated in favor of other toolkits
* a direct reliance on Eigen, which introduces heavy templating and a limited GPU support

CTranslate2 addresses these issues in several ways:

* the core implementation is framework agnostic, moving the framework specific logic to a model conversion step
* the internal operators follow the ONNX specifications as much as possible for better future-proofing
* the call to external libraries (Intel MKL, cuBLAS, etc.) occurs as late as possible in the execution to not rely on a library specific logic

### What is the state of this project?

The code has been generously tested in production settings so people can rely on it in their application. The following APIs are covered by backward compatibility guarantees:

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
* You need to embed translation models in an existing C++ application.
* Your application requires custom threading and memory usage control.
* You want to reduce the model size on disk and/or memory.

However, you should probably **not** use this project when:

* You want to train custom architectures not covered by this project.
* You see no value in the key features listed at the top of this document.

### What are the known limitations?

The current approach only exports the weights from existing models and redefines the computation graph via the code. This implies a strong assumption of the graph architecture executed by the original framework.

We are actively looking to ease this assumption by supporting ONNX as model parts.

### What are the future plans?

There are many ways to make this project better and faster. See the open issues for an overview of current and planned features. Here are some things we would like to get to:

* Better support of INT8 quantization, for example by quantizing more layers
* Support of running ONNX graphs

### What is the difference between `intra_threads` and `inter_threads`?

* `intra_threads` is the number of threads that is used within operators: increase this value to decrease the latency.
* `inter_threads` is the maximum number of translations executed in parallel: increase this value to increase the throughput (this will also increase the memory usage as some internal buffers are duplicated for thread safety)

The total number of computing threads launched by the process is summarized by this formula:

```text
num_threads = inter_threads * intra_threads
```

Note that these options are only defined for CPU translation. In particular, `inter_threads` is forced to 1 when executing on GPU.

### Do you provide a translation server?

There is currently no translation server. We may provide a basic server in the future but we think it is up to the users to serve the translation depending on their requirements.

### How can I generate a vocabulary mapping file?

See [here](https://github.com/OpenNMT/papers/tree/master/WNMT2018/vmap).
