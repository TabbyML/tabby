# CTranslate2

CTranslate2 is a custom inference engine for neural machine translation models supporting both CPU and GPU execution. It is:

* **fast**, up to 2x times faster than an already competitive OpenNMT-py translation;
* **portable**, no architecture-specific flags are required during the compilation;
* **lightweight**, the CPU library takes about 30MB with its dependencies.

The project currently provides converters for Transformer models trained with OpenNMT-tf and OpenNMT-py and exposes translation APIs in Python and C++.

## Benchmark

*TODO*

## Converting models

A model conversion step is required to transform trained models into the CTranslate2 representation (see available converters in `python/ctranslate2/converters`). To get you started, here are the command lines to convert pre-trained OpenNMT-tf and OpenNMT-py models with INT16 quantization:

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
| float             | float         | float            | float |

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

translator.initialize(4)  # Number of MKL threads.

t = translator.Translator(
    model_path: str      # Path to the CTranslate2 model directory.
    device="cpu",        # Can be "cpu", "cuda", or "auto".
    thread_pool_size=2)  # Number of concurrent translations.

output = t.translate_batch(
    tokens: list,            # A list of list of string.
    beam_size=4,             # Beam size.
    num_hypotheses=1,        # Number of hypotheses to return.
    length_penalty=0.6,      # Length penalty constant.
    max_decoding_steps=250,  # Maximum decoding steps.
    min_decoding_length=1,   # Minimum prediction length (EOS excluded).
    use_vmap=False)          # Use the VMAP saved in this model.

# output is a 2D list [batch x num_hypotheses] containing tuples of (score, tokens).

del t  # Release translator resources.
```

### With the C++ API

```cpp
#include <ctranslate2/translator.h>

int main() {
  auto model = ctranslate2::models::ModelFactory::load("ende_ctranslate2", ctranslate2::Device::CUDA);

  std::vector<std::string> input{"Hello", "world", "!"};
  std::vector<std::string> output;

  ctranslate2::Translator translator(model);
  ctranslate2::TranslationOptions options;
  ctranslate2::TranslationResult result = translator.translate(input, options);

  output = result.output();
  return 0;
}
```

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

The code has been generously tested in production settings so people can rely on it in their application. The following APIs are covered by backward compatibility guarantees:

* Converted models
* Python modules:
  * `ctranslate2.Translator`
  * `ctranslate2.converters.OpenNMTPyConverter`
  * `ctranslate2.converters.OpenNMTTFConverter`
* C++ symbols:
  * `ctranslate2::init`
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

However, you should probably **not** use this project when:

* You want to train custom architectures not covered by this project.
* Your only validation metric is the BLEU score.

### What are the known limitations?

The current approach only exports the weights from existing models and redefines the computation graph via the code. This implies a strong assumption of the graph architecture executed by the original framework.

We are actively looking to ease this assumption by supporting ONNX as model parts.

### What are the future plans?

There are many ways to make this project better and faster. See the open issues for an overview of current and planned features. Here are some things we would like to get to:

* INT8 quantization for CPU and GPU
* Support of running ONNX graphs
* More configurable memory cache

### What is the difference between `intra_threads` and `inter_threads`?

* `intra_threads` is the number of threads that is used within operators: increase this value to decrease the latency for CPU translation.
* `inter_threads` is the maximum number of translations executed in parallel: increase this value to increase the throughput.

The total number of threads launched by the process is summarized by this formula:

```text
num_threads = inter_threads * min(intra_threads, num_cores)
```

Some notes about `inter_threads`:

* On GPU, this value is forced to 1 as the code is not yet synchronization-free
* Increasing this value also increases the memory usage as internal buffers has to be separate

### Do you provide a translation server?

There is currently no translation server. We may provide a basic server in the future but we think it is up to the users to serve the translation depending on their requirements.

### How can I generate a vocabulary mapping file?

See [here](https://github.com/OpenNMT/papers/tree/master/WNMT2018/vmap).
