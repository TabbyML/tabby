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

## Tool to tune intra_threads and inter_threads

You can use the script `tune_inter_intra.py` to find the best values for your hardware.
Simply replace your call to `./build/cli/translate` by `python3 ./tools/tune_inter_intra.py ./build/cli/translate`.
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
