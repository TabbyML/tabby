# Performance

## Improving performance

Below are some recommendations to further improve translation performance. Many of these recommendations were used in the [WNGT 2020 efficiency task submission](../examples/wngt2020).

### General

* Set the compute type to "auto" to automatically select the fastest execution path on the current system
* Reduce the beam size to the minimum value that meets your quality requirement
* When using a beam size of 1, keep `return_scores` disabled if you are not using prediction scores: the final softmax layer can be skipped
* Set `max_batch_size` and pass a larger batch to `translate_batch`: the input sentences will be sorted by length and split by chunk of `max_batch_size` elements for improved efficiency
* Prefer the "tokens" `batch_type` to make the total number of elements in a batch more constant

### CPU

* Use an Intel CPU supporting AVX512
* If you are translating a large volume of data, prefer increasing `inter_threads` over `intra_threads` to improve scalability
* Avoid the total number of threads `inter_threads * intra_threads` to be larger than the number of physical cores

### GPU

* Use a NVIDIA GPU with Tensor Cores (Compute Capability >= 7.0)
* Pass multiple GPU IDs to `device_index` to run translations on multiple GPUs

## Measuring performance

### Reporting the translation throughput

The command line option `--log_throughput` reports the *tokens generated per second* on the standard error output. This is the recommended metric to compare different runs (higher is better).

### Benchmarking models

See the [benchmark scripts](../tools/benchmark).

### Profiling the execution

The command line option `--log_profiling` reports an execution profile on the standard error output. It prints a list of selected functions in the format:

```text
  2.51%  80.38%  87.27% beam_search                 557.00ms
```

where the columns mean:

1. Percent of time spent in the function
2. Percent of time spent in the function and its callees
3. Percent of time printed so far
4. Name of the function
5. Time spent in the function (in milliseconds)

The list is ordered on 5. from the largest to smallest time.

## GPU performance

### CUDA caching allocator

Allocating memory on the GPU with `cudaMalloc` is costly and is best avoided in high-performance code. For this reason CTranslate2 integrates caching allocators which enable a fast reuse of previously allocated buffers. The allocator can be selected with the environment variable `CT2_CUDA_ALLOCATOR`:

#### `cub_caching` (default)

This caching allocator from the [CUB project](https://github.com/NVIDIA/cub) can be tuned to tradeoff memory usage and speed. By default, CTranslate2 uses the following values which have been selected experimentally:

* `bin_growth = 4`
* `min_bin = 3`
* `max_bin = 12`
* `max_cached_bytes = 209715200` (200MB)

You can override these values by setting the environment variable `CT2_CUDA_CACHING_ALLOCATOR_CONFIG` with comma-separated values in the same order as the list above:

```bash
export CT2_CUDA_CACHING_ALLOCATOR_CONFIG=8,3,7,6291455
```

See the description of each value in the [allocator implementation](https://github.com/NVIDIA/cub/blob/main/cub/util_allocator.cuh).

#### `cuda_malloc_async`

CUDA 11.2 introduced an [asynchronous allocator with memory pools](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html). It is usually faster than `cub_caching` but uses more memory.

## CPU performance

### Packed GEMM

Packed GEMM could improve performance for single-core decoding. You can enable this mode by setting the environment variable `CT2_USE_EXPERIMENTAL_PACKED_GEMM=1`. See [Intel's article](https://software.intel.com/content/www/us/en/develop/articles/introducing-the-new-packed-apis-for-gemm.html) to learn more about packed GEMM.

### Tuning `intra_threads` and `inter_threads`

You can use the script `tools/tune_inter_intra.py` to find the threading configuration that maximizes the global throughput.
Simply replace your call to `./build/cli/translate` by `python3 ./tools/tune_inter_intra.py ./build/cli/translate`. The script will run the translation multiple times and report the final tokens per second metric and the maximum memory usage for each threading combination.

```bash
head -n 100 valid.de | python3 ./tools/tune_inter_intra.py ./build/cli/translate --model ende_ctranslate2 --beam_size 2 > values.csv
column -s, -t < out.csv | sort -k3 -r
```
```bash
inter_threads  intra_threads  tokens/s  memory_used (in MB)
4              2              919.333   918
2              4              919.333   706
1              8              919.333   557
8              1              689.5     914
7              1              689.5     910
3              2              689.5     876
2              3              689.5     731
2              2              689.5     729
1              5              689.5     562
1              7              689.5     553
1              4              689.5     553
1              6              689.5     549
5              1              551.6     914
4              1              551.6     910
6              1              551.6     869
3              1              551.6     861
1              3              551.6     567
2              1              394.0     715
1              2              394.0     562
1              1              212.154   559
```
