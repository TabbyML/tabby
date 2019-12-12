# Performance

## Measuring performance

### Throughput

The command line option `--log_throughput` reports *tokens generated per second* on the standard error output. This is the recommended metric to compare different runs (higher is better).

### Profiling

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

## CUDA caching allocator

Allocating memory on the GPU with `cudaMalloc` is costly and is best avoided in high-performance code. For this reason CTranslate2 uses a [caching allocator](https://nvlabs.github.io/cub/structcub_1_1_caching_device_allocator.html) which enables a fast reuse of previously allocated buffers.

The caching allocator can be tuned to tradeoff memory usage and speed (see the description in the link above). By default, CTranslate2 uses the following values which have been selected experimentally:

* `bin_growth = 4`
* `min_bin = 3`
* `max_bin = 12`
* `max_cached_bytes = 209715200` (200MB)

You can override these values by setting the environment variable `CT2_CUDA_CACHING_ALLOCATOR_CONFIG` with comma-separated values in the same order as the list above:

```bash
export CT2_CUDA_CACHING_ALLOCATOR_CONFIG=8,3,7,6291455
```
