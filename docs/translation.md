# Text translation

CTranslate2 exposes high-level classes to run text translation from Python and C++. The main entrypoint in Python is the [`Translator`](python/ctranslate2.Translator.rst) class which provides methods to translate files or batches as well as methods to score existing translations.

## Examples

Here are some translation examples using the model converted in the [quickstart](quickstart.md).

### Python

```python
import ctranslate2

translator = ctranslate2.Translator("ende_ctranslate2/", device="cpu")
results = translator.translate_batch([["▁H", "ello", "▁world", "!"]])

print(results[0].hypotheses[0])
```

See the [`Translator`](python/ctranslate2.Translator.rst) class documentation for available options.

### C++

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

See the [`TranslatorPool`](https://github.com/OpenNMT/CTranslate2/blob/master/include/ctranslate2/translator_pool.h) class for more advanced usages such as asynchronous translations.

### Translation client

The translation client can be used via the Docker image:

```bash
echo "▁H ello ▁world !" | docker run -i --rm -v $PWD:/data \
    opennmt/ctranslate2:latest-ubuntu20.04-cuda11.2 --model /data/ende_ctranslate2 --device cpu
```

To translate on GPU, use `docker run --gpus all` and set the option `--device cuda`. Use `--help` to see the list of available options.

## Source factors

Models using additional source factors (a.k.a. source features) are supported. The factors should be added directly to the source input tokens using the special separator ￨ in both the file and batch translations APIs.

In API:

```python
translator.translate_batch([["hello￨C", "world￨L", "!￨N"]])
```

In text file:

```text
hello￨C world￨L !￨N
```

An error is raised if the number of factors does not match what the model expects.

## Dynamic vocabulary reduction

The target vocabulary can be dynamically reduced to increase the translation speed. A source-target vocabulary mapping file (a.k.a. *vmap*) should be passed to the converter option `--vocab_mapping` and enabled during translation with `use_vmap`.

The vocabulary mapping file maps source N-grams to a list of candidate target tokens. During translation, the target vocabulary will be dynamically reduced to the union of all target tokens associated with the N-grams from the batch to translate.

It is a text file where each line has the following format:

```text
src_1 src_2 ... src_N<TAB>tgt_1 tgt_2 ... tgt_K
```

If the source N-gram is empty (N = 0), the assiocated target tokens will always be included in the reduced vocabulary.

```{hint}
See [here](https://github.com/OpenNMT/papers/tree/master/WNMT2018/vmap) for an example on how to generate this file.
```
