# Multithreading and parallelism

CTranslate2 has 2 level of parallelization:

* `inter_threads` which is the maximum number of batches executed in parallel.<br/>**=> Increase this value to increase the throughput.**
* `intra_threads` which is the number of OpenMP threads that is used per batch.<br/>**=> Increase this value to decrease the latency on CPU.**

The total number of computing threads launched by the process is `inter_threads * intra_threads`.

```{note}
Even though the model data are shared between parallel replicas, increasing `inter_threads` will still increase the memory usage as some internal buffers are duplicated for thread safety.
```

On GPU, batches processed in parallel are using separate CUDA streams. Depending on the workload and GPU specifications this may or may not improve the global throughput. For better parallelism on GPU, consider using multiple GPUs as described below.

## Parallel execution

The [`Translator`](python/ctranslate2.Translator.rst) and [`Generator`](python/ctranslate2.Generator.rst) instances can be configured to process multiple batches in parallel, including on multiple GPUs:

```python
# Create a CPU translator with 4 workers each using 1 thread:
translator = ctranslate2.Translator(model_path, device="cpu", inter_threads=4, intra_threads=1)

# Create a GPU translator with 4 workers each running on a separate GPU:
translator = ctranslate2.Translator(model_path, device="cuda", device_index=[0, 1, 2, 3])

# Create a GPU translator with 4 workers each using a different CUDA stream:
translator = ctranslate2.Translator(model_path, device="cuda", inter_threads=4)
```

Multiple batches should be submitted concurrently to enable this parallelization. Parallel translations are enabled in the following cases:

* When calling `{translate,score}_file`
* When calling `{translate,score,generate}_iterable`
* When calling `{translate,score,generate}_batch` and setting `max_batch_size`: the input will be split according to `max_batch_size` and each sub-batch will be translated in parallel.
* When calling `{translate,score,generate}_batch` from multiple Python threads.
* When calling `{translate,score,generate}_batch` multiple times with `asynchronous=True`.

```{note}
Parallelization with multiple Python threads is possible because all computation methods release the [Python GIL](https://wiki.python.org/moin/GlobalInterpreterLock).
```

## Asynchronous execution

The methods `translate_batch`, `score_batch`, and `generate_batch` can run asynchronously with `asynchronous=True`. In this mode, the method returns immediately and the result can be retrieved later:

```python
async_results = []
for batch in batch_generator():
    async_results.extend(translator.translate_batch(batch, asynchronous=True))

for async_result in async_results:
    print(async_result.result())  # This method blocks until the result is available.
```

```{attention}
The `Translator` and `Generator` objects have a limited queue size by default. When the queue of batches is full, the method will block even with `asynchronous=True`. See the parameter `max_queued_batches` in their constructor to configure the queue size.
```
