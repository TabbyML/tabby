# Multithreading and parallelism

## Intra-op multithreading on CPU

Most model operations (matmul, softmax, etc.) are using multiple threads on CPU. The number of threads can be configured with the parameter `intra_threads` (the default value is 4):

```python
translator = ctranslate2.Translator(model_path, device="cpu", intra_threads=8)
```

This multithreading is generally implemented with [OpenMP](https://www.openmp.org/) so the threads behavior can also be customized with the different `OMP_*` environment variables.

When OpenMP is disabled (which is the case for example in the Python ARM64 wheels for macOS), the multithreading is implemented with [`BS::thread_pool`](https://github.com/bshoshany/thread-pool).

## Data parallelism

Objects running models such as the [`Translator`](python/ctranslate2.Translator.rst) and [`Generator`](python/ctranslate2.Generator.rst) can be configured to process multiple batches in parallel, including on multiple GPUs:

```python
# Create a CPU translator with 4 workers each using 1 intra-op thread:
translator = ctranslate2.Translator(model_path, device="cpu", inter_threads=4, intra_threads=1)

# Create a GPU translator with 4 workers each running on a separate GPU:
translator = ctranslate2.Translator(model_path, device="cuda", device_index=[0, 1, 2, 3])

# Create a GPU translator with 4 workers each using a different CUDA stream:
# (Note: depending on the workload and GPU specifications this may not improve the global throughput.)
translator = ctranslate2.Translator(model_path, device="cuda", inter_threads=4)
```

When the workers are running on the same device, the model weights are shared to save on memory.

Multiple batches should be submitted concurrently to enable this parallelization. Parallel executions are enabled in the following cases:

* When calling methods from multiple Python threads.
* When calling methods multiple times with `asynchronous=True`.
* When calling file-based or stream-based methods.
* When setting `max_batch_size`: the input will be split according to `max_batch_size` and each sub-batch will be executed in parallel.

```{note}
Parallelization with multiple Python threads is possible because all computation methods release the [Python GIL](https://wiki.python.org/moin/GlobalInterpreterLock).
```

## Model and tensor parallelism

These types of parallelism are not yet implemented in CTranslate2.

## Asynchronous execution

Some methods can run asynchronously with `asynchronous=True`. In this mode, the method returns immediately and the result can be retrieved later:

```python
async_results = []
for batch in batch_generator():
    async_results.extend(translator.translate_batch(batch, asynchronous=True))

for async_result in async_results:
    print(async_result.result())  # This method blocks until the result is available.
```

```{attention}
Instances supporting asynchronous execution have a limited queue size by default. When the queue of batches is full, the method will block even with `asynchronous=True`. See the parameter `max_queued_batches` in their constructor to configure the queue size.
```
