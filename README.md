# CTranslate2

CTranslate2 is a custom inference engine for neural machine translation models supporting both CPU and GPU execution.

## Key features

* **Fast execution**<br/>The execution aims to be faster than a general purpose deep learning framework: on standard translation tasks, it is [up to 3x faster](#benchmarks) than OpenNMT-py.
* **Model quantization**<br/>Support INT16 quantization on CPU and INT8 quantization (experimental) on CPU and GPU.
* **Parallel translation**<br/>Translations can be run efficiently in parallel without duplicating the model data in memory.
* **Dynamic memory usage**<br/>The memory usage changes dynamically depending on the request size while still meeting performance requirements thanks to caching allocators on both CPU and GPU.
* **Automatic instruction set dispatch**<br/>When using Intel MKL, the dispatch to the optimal instruction set is done at runtime.
* **Ligthweight on disk**<br/>Models can be quantized below 100MB with minimal accuracy loss. A full featured Docker image supporting GPU and CPU requires less than 1GB.
* **Easy to use translation APIs**<br/>The project exposes [translation APIs](#translating) in Python and C++ to cover most integration needs.

Some of these features are difficult to achieve in standard deep learning frameworks and are the motivation for this project.

### Supported decoding options

The translation API supports several decoding options:

* decoding with greedy or beam search
* translating with a target prefix
* constraining the decoding length
* returning multiple translation hypotheses
* returning attention vectors
* approximating the generation using a pre-compiled [vocabulary map](#how-can-i-generate-a-vocabulary-mapping-file)

## Dependencies

CTranslate2 uses the following libraries for acceleration:

* CPU
  * [Intel MKL](https://software.intel.com/en-us/mkl)
  * [Intel MKL-DNN](https://github.com/intel/mkl-dnn)
* GPU
  * [CUB](https://nvlabs.github.io/cub/)
  * [TensorRT](https://developer.nvidia.com/tensorrt)
  * [Thrust](https://docs.nvidia.com/cuda/thrust/index.html)
  * [cuBLAS](https://developer.nvidia.com/cublas)
  * [cuDNN](https://developer.nvidia.com/cudnn)

## Converting models

A model conversion step is required to transform trained models into the CTranslate2 representation. The following frameworks and models are currently supported:

|     | [OpenNMT-tf](python/ctranslate2/converters/opennmt_tf.py) | [OpenNMT-py](python/ctranslate2/converters/opennmt_py.py) |
| --- | --- | --- |
| TransformerBase | Yes | Yes |
| TransformerBig  | Yes | Yes |

To get you started, here are the command lines to convert pre-trained OpenNMT-tf and OpenNMT-py models with int16 quantization:

### OpenNMT-tf

```bash
cd python/

wget https://s3.amazonaws.com/opennmt-models/averaged-ende-export500k.tar.gz
tar xf averaged-ende-export500k.tar.gz

python -m ctranslate2.converters.opennmt_tf \
    --model_dir averaged-ende-export500k/1554540232/ \
    --output_dir ende_ctranslate2 \
    --model_spec TransformerBase \
    --quantization int16
```

### OpenNMT-py

```bash
cd python/

wget https://s3.amazonaws.com/opennmt-models/transformer-ende-wmt-pyOnmt.tar.gz
tar xf transformer-ende-wmt-pyOnmt.tar.gz

python -m ctranslate2.converters.opennmt_py \
    --model_path averaged-10-epoch.pt \
    --output_dir ende_ctranslate2 \
    --model_spec TransformerBase \
    --quantization int16
```

### Adding converters

Each converter should populate a model specification with trained weights coming from an existing model. The model specification declares the variable names and layout expected by the CTranslate2 core engine.

See the existing converters implementation which could be used as a template.

### Quantization

The converters support model quantization which is a way to reduce the model size and accelerate its execution. However, some execution settings are not (yet) optimized for all quantization types. The following table documents the actual types used during the computation:

| Model type | GPU   | CPU (AVX2) | CPU (older) |
| ---------- | ----- | ---------- | ----------  |
| int16      | float | int16      | float       |
| int8       | int8  | int8       | float       |

**Notes:**

* only GEMM-based layers and embeddings are currently quantized

## Building

### Docker

See the `docker/` directory.

### Binaries (Ubuntu)

The minimum requirements for building CTranslate2 binaries are Intel MKL and `libboost-program-options-dev`.

**Note:** This minimal installation only enables CPU execution. For GPU support, see how the [GPU Dockerfile](docker/Dockerfile.centos7-gpu) is defined.

#### Install MKL

Use the following instructions to install MKL:

```bash
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
sudo sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
sudo apt-get update
sudo apt-get install intel-mkl-64bit-2019.4-070
```

Go to https://software.intel.com/en-us/articles/installing-intel-free-libs-and-python-apt-repo for more detail.

#### Install libboost-program-options-dev

```bash
sudo apt-get install libboost-program-options-dev
```

#### Compile

Under the project root then launch the following commands:

```bash
mkdir build && cd build
cmake ..
make -j4
```

The binary `translate` will be generated in the directory `cli`:

```bash
echo "▁H ello ▁world !" | ./cli/translate --model ../python/ende_ctranslate2/
```

The result `▁Hallo ▁Welt !` should be display.

**Note:** Before running the command, you should get your model by following the [Converting models](#converting-models) section.

## Translating

Docker images are currently the recommended way to use the project as they embeds all dependencies and are optimized.

The library has several entrypoints which are briefly introduced below. The examples use the English-German model downloaded in [Converting models](#converting-models) which requires a SentencePiece tokenization.

### With the translation client

```bash
echo "▁H ello ▁world !" | docker run -i --rm -v $PWD:/data \
    systran/ctranslate2 --model /data/ende_ctranslate2
```

*See `docker run --rm systran/ctranslate2 --help` for additional options.*

### With the Python API

```python
from ctranslate2 import translator
t = translator.Translator("ende_ctranslate2/")

input_tokens = ["▁H", "ello", "▁world", "!"]
result = t.translate_batch([input_tokens])

print(result[0][0])
```

*See the [Python reference](docs/python.md) for more advanced usage.*

### With the C++ API

```cpp
#include <iostream>
#include <ctranslate2/translator.h>

int main() {
  ctranslate2::Translator translator("ende_ctranslate2/", ctranslate2::Device::CPU);
  ctranslate2::TranslationResult result = translator.translate({"▁H", "ello", "▁world", "!"});

  for (const auto& token : result.output())
    std::cout << token << ' ';
  std::cout << std::endl;
  return 0;
}
```

*See the [Translator class](include/ctranslate2/translator.h) for more advanced usage, and the [TranslatorPool class](include/ctranslate2/translator_pool.h) for running translations in parallel.*

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

Then follow the CTranslate2 [build instructions](#building) again to produce the test executable `tests/ctranslate2_test`. The binary expects the path to the test data as argument:

```bash
./tests/ctranslate2_test ../tests/data
```

## Benchmarks

We compare OpenNMT-py and CTranslate2 on the [English-German base Transformer model](http://opennmt.net/Models-py/#translation).

### Model size

| | Model size |
| --- | --- |
| CTranslate2 (int8) | 110MB |
| CTranslate2 (int16) | 197MB |
| CTranslate2 (float) | 374MB |
| OpenNMT-py | 542MB |

CTranslate2 models are generally lighter and can go as low as 100MB when quantized to int8. This also results in a fast loading time and noticeable lower memory usage during runtime.

### Speed

We translate the test set *newstest2014* and report:

* the number of target tokens generated per second (higher is better)
* the BLEU score of the detokenized output (higher is better)

Unless otherwise noted, translations are running beam search with a size of 4 and a maximum batch size of 32. Please note that the results presented below are only valid for the configuration used during this benchmark: absolute and relative performance may change with different settings.

#### GPU

Configuration:

* **GPU:** NVIDIA Tesla V100
* **CUDA:** 10.0
* **Driver:** 410.48

| | Tokens/s | BLEU |
| --- | --- | --- |
| CTranslate2 0.16.4 | 3077.89 | 26.69 |
| CTranslate2 0.16.4 (int8) | 1822.21 | 26.79 |
| OpenNMT-py 0.9.2 | 980.44 | 26.69 |

*Note on int8: it is slower as the runtime overhead of int8<->float conversions is presently too high. However, it results in a much lower memory usage and also acts as a regularizer (hence the higher BLEU score in this case).*

#### CPU

Configuration:

* **CPU:** Intel(R) Core(TM) i7-6700K CPU @ 4.00GHz (with AVX2)
* **Number of threads:** 4 "intra threads", 1 "inter threads" ([what's the difference?](#what-is-the-difference-between-intra_threads-and-inter_threads))

| | Tokens/s | BLEU |
| --- | --- | --- |
| CTranslate2 0.16.4 (int16 + vmap) | 419.47 | 26.63 |
| CTranslate2 0.16.4 (int8) | 339.46 | 26.69 |
| CTranslate2 0.16.4 (int16) | 339.43 | 26.68 |
| CTranslate2 0.16.4 (float) | 335.34 | 26.69 |
| OpenNMT-py 0.9.2 | 241.92 | 26.69 |

*Note on quantization: higher performance gains over float are expected with lower beam size and/or intra threads.*

We are also interested in comparing the performance of a minimal run: single thread, single example in the batch, and greedy search. This could be a possible setting in a constrained environment.

| | Tokens/s |
| --- | --- |
| CTranslate2 0.16.4 (int8) | 133.18 |
| CTranslate2 0.16.4 (int16) | 130.43 |
| CTranslate2 0.16.4 (float) | 112.66 |
| OpenNMT-py 0.9.2 | 80.88 |

### Memory usage

We don't have numbers comparing memory usage yet. However, past experiments showed that CTranslate2 usually requires up to 2x less memory than OpenNMT-py.

## FAQ

### How does it relate to the original [CTranslate](https://github.com/OpenNMT/CTranslate) project?

The original *CTranslate* project shares a similar goal which is to provide a custom execution engine for OpenNMT models that is lightweight and fast. However, it has some limitations that were hard to overcome:

* a strong dependency on LuaTorch and OpenNMT-lua, which are now both deprecated in favor of other toolkits
* a direct reliance on Eigen, which introduces heavy templating and a limited GPU support

CTranslate2 addresses these issues in several ways:

* the core implementation is framework agnostic, moving the framework specific logic to a model conversion step
* the internal operators follow the ONNX specifications as much as possible for better future-proofing
* the call to external libraries (Intel MKL, cuBLAS, etc.) occurs as late as possible in the execution to not rely on a library specific logic

### What is the state of this project?

The code has been generously tested in production settings so people can rely on it in their application. The following APIs are covered by backward compatibility guarantees (enforced after the 1.0 release):

* Converted models
* Python symbols:
  * `ctranslate2.Translator`
  * `ctranslate2.converters.OpenNMTPyConverter`
  * `ctranslate2.converters.OpenNMTTFConverter`
* C++ symbols:
  * `ctranslate2::ModelFactory`
  * `ctranslate2::TranslationOptions`
  * `ctranslate2::TranslationResult`
  * `ctranslate2::Translator`
  * `ctranslate2::TranslatorPool`

Other APIs are expected to evolve to increase efficiency, genericity, and model support.

### Why and when should I use this implementation instead of PyTorch or TensorFlow?

Here are some scenarios where this project could be used:

* You want to accelarate standard translation models for production usage, especially on CPUs.
* You need to embed translation models in an existing C++ application.
* You need portable binaries that automatically dispatch the execution to the best instruction set.
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
