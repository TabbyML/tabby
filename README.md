# CTranslate2

CTranslate2 is a custom inference engine for neural machine translation models supporting both CPU and GPU execution.

## Key features

* **Fast execution**<br/>The execution aims to be faster than a general purpose deep learning framework, especially on CPU which is one of the main motivation for this project.
* **Parallel translation**<br/>Translations can be run efficiently in parallel without duplicating the model data in memory.
* **Dynamic memory usage**<br/>The memory usage changes dynamically depending on the request size while still meeting performance requirements thanks to caching allocators on both CPU and GPU.
* **Portable binary**<br/>The compilation does not require a target instruction set, the dispatch is done at runtime.
* **Ligthweight on disk**<br/>The CPU library takes about 30MB on disk with its dependencies and models can be compressed below 100MB with minimal accuracy loss. A full featured GPU Docker image requires less than 800MB.
* **Easy to use translation APIs**<br/>The project exposes [translation APIs](#translating) in Python and C++ to cover most integration needs.

Some of these features are difficult to achieve in standard deep learning frameworks and are the motivation for this project.

### Supported decoding options

The translation API supports several decoding options:

* decoding with greedy or beam search
* constraining the decoding length
* returning multiple translation hypotheses
* returning attention vectors
* approximating the generation using a pre-compiled [vocabulary map](##how-can-i-generate-a-vocabulary-mapping-file)

## Dependencies

CTranslate2 uses the following libraries for acceleration:

* CPU
  * [Intel MKL](https://software.intel.com/en-us/mkl)
* GPU
  * [cuBLAS](https://developer.nvidia.com/cublas)
  * [TensorRT](https://developer.nvidia.com/tensorrt)
  * [CUB](https://nvlabs.github.io/cub/)

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
    --model_dir averaged-ende-export500k/1539080952/ \
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

The converters support model quantization which is a way to reduce the model size and accelerate its execution. However, some execution settings are not (yet) optimized for all quantization types. The following table document the actual types used during the model execution:

| Model type        | CPU with AVX2 | CPU without AVX2 | GPU   |
| ----------------- | ------------- | ---------------- | ----- |
| int8              | int16         | float            | float |
| int16             | int16         | float            | float |

## Translating

Docker images are currently the recommended way to use the project as they embeds all dependencies and are optimized.

```bash
docker pull systran/ctranslate2_gpu:latest
```

The library has several entrypoints which are briefly introduced below:

### With the translation client

```bash
nvidia-docker run -it --rm -v $PWD/my_data:/data \
    --entrypoint /root/ctranslate2/bin/translate \
    systran/ctranslate2_gpu:latest \
    --model /data/ende_ctranslate2 \
    --src /data/newstest2014.en \
    --batch_size 32 \
    --beam_size 2
```

### With the Python API

```python
from ctranslate2 import translator
t = translator.Translator("my_data/ende_ctranslate2/")

input_tokens = ["Hello", "world", "!"]
result = t.translate_batch([input_tokens])

print(result[0][0])
```

*See the [Python reference](python/README.md) for more advanced usage.*

### With the C++ API

```cpp
#include <iostream>
#include <ctranslate2/translator.h>

int main() {
  ctranslate2::Translator translator("my_data/ende_ctranslate2/", ctranslate2::Device::CPU);
  ctranslate2::TranslationResult result = translator.translate({"Hello", "world", "!"});

  for (const auto& token : result.output())
    std::cout << token << ' ';
  std::cout << std::endl;
  return 0;
}
```

*See the [Translator class](include/ctranslate2/translator.h) for more advanced usage, and the [TranslatorPool class](include/ctranslate2/translator_pool.h) for running translations in parallel.*

## Benchmark

This benchmark compares standard OpenNMT-py and CTranslate2 runs on the same English-German Transformer model.

* Test file: `sacrebleu -t wmt14 -l en-de --echo src` (2737 lines)
* BLEU signature: `BLEU+case.mixed+lang.en-de+numrefs.1+smooth.exp+test.wmt14+tok.13a+version.1.2.12`
* Translation options:
  * batch size: 32
  * beam size: 4

### CPU

| Model               | Size  | Time  | Memory | BLEU |
| ------------------- | ----- | ----  | -----  | ---- |
| OpenNMT-py          | 542MB | 519 s | 1.3GB  | 26.8 |
| CTranslate2         | 392MB | 288 s | 0.8GB  | 26.8 |
| + *int16*           | 207MB | 266 s | 0.7GB  | 26.8 |
| + *vmap*            | 219MB | 216 s | 0.6GB  | 26.7 |

(Measured on a *Intel(R) Core(TM) i7-6700K* CPU and 4 threads, memory usage is approximated and rounded to the nearest 0.1GB.)

### GPU

| Model               | Size  | Time | BLEU |
| ------------------- | ----- | ---- | ---- |
| OpenNMT-py          | 542MB | 92 s | 26.8 |
| CTranslate2         | 392MB | 31 s | 26.8 |

(Measured on a *p3.2xlarge* EC2 instance.)

## Building

To build the Docker images, some external dependencies need to be downloaded separately and placed in the `deps/` directory:

1. [Intel MKL](https://software.intel.com/en-us/mkl/choose-download/linux)
2. [TensorRT](https://developer.nvidia.com/tensorrt)

```bash
docker build -t systran/ctranslate2 -f Dockerfile .
docker build -t systran/ctranslate2_gpu -f Dockerfile.cuda .
```

For complete compilation instructions, see the *Dockerfiles*.

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
* You need to embed translation models in an existing application without adding large dependencies.
* You need portable binaries that automatically dispatch the execution to the best instruction set.
* Your application requires custom threading and memory usage control.

However, you should probably **not** use this project when:

* You want to train custom architectures not covered by this project.
* You see no value in the key features listed at the top of this document.
* Your only validation metric is the BLEU score.

### What are the known limitations?

The current approach only exports the weights from existing models and redefines the computation graph via the code. This implies a strong assumption of the graph architecture executed by the original framework.

We are actively looking to ease this assumption by supporting ONNX as model parts.

### What are the future plans?

There are many ways to make this project better and faster. See the open issues for an overview of current and planned features. Here are some things we would like to get to:

* INT8 quantization for CPU and GPU
* Support of running ONNX graphs

### What is the difference between `intra_threads` and `inter_threads`?

* `intra_threads` is the number of threads that is used within operators: increase this value to decrease the latency for CPU translation.
* `inter_threads` is the maximum number of translations executed in parallel: increase this value to increase the throughput.

The total number of computing threads launched by the process is summarized by this formula:

```text
num_threads = inter_threads * min(intra_threads, num_physical_cores)
```

Some notes about `inter_threads`:

* On GPU, this value is forced to 1 as the code is not yet synchronization-free
* Increasing this value also increases the memory usage as some internal buffers are duplicated for thread safety

### Do you provide a translation server?

There is currently no translation server. We may provide a basic server in the future but we think it is up to the users to serve the translation depending on their requirements.

### How can I generate a vocabulary mapping file?

See [here](https://github.com/OpenNMT/papers/tree/master/WNMT2018/vmap).
