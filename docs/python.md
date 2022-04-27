# Python

* [Quickstart](../README.md#quickstart)
* [Installation](../README.md#installation)
* [API reference](https://opennmt.net/CTranslate2)

## Parallel and asynchronous execution

The [`Translator`](https://opennmt.net/CTranslate2/src/ctranslate2.Translator.html) and [`Generator`](https://opennmt.net/CTranslate2/src/ctranslate2.Generator.html) instances can be configured to process multiple batches in parallel:

```python
# Create a CPU translator with 4 workers each using 1 thread:
translator = ctranslate2.Translator(model_path, device="cpu", inter_threads=4, intra_threads=1)

# Create a GPU translator with 4 workers each running on a separate GPU:
translator = ctranslate2.Translator(model_path, device="cuda", device_index=[0, 1, 2, 3])

# Create a GPU translator with 4 workers each using a different CUDA stream:
translator = ctranslate2.Translator(model_path, device="cuda", inter_threads=4)
```

Parallel translations are enabled in the following cases:

* When calling `{translate,score}_file`.
* When calling `{translate,score,generate}_batch` and setting `max_batch_size`: the input will be split according to `max_batch_size` and each sub-batch will be translated in parallel.
* When calling `{translate,score,generate}_batch` from multiple Python threads: parallelization with Python threads is made possible because the `Translator` methods release the [Python GIL](https://wiki.python.org/moin/GlobalInterpreterLock).
* When calling `{translate,generate}_batch` multiple times with `asynchronous=True`:

```python
async_results = []
for batch in batch_generator():
    async_results.extend(translator.translate_batch(batch, asynchronous=True))

for async_result in async_results:
    print(async_result.result())  # This method blocks until the result is available.
```

## Memory management

To release the resources used by `Translator` and `Generator` instances, you can simply delete the object, e.g.:

```python
del translator
```

In some cases, you might want to temporarily unload the model and load it back later. The `Translator` object provides the methods [`unload_model`](https://opennmt.net/CTranslate2/src/ctranslate2.Translator.html#ctranslate2.Translator.unload_model) and [`load_model`](https://opennmt.net/CTranslate2/src/ctranslate2.Translator.html#ctranslate2.Translator.load_model) for this purpose.
