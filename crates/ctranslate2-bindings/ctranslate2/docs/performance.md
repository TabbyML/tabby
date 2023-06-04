# Performance tips

Below are some general recommendations to further improve performance. Many of these recommendations were used in the [WNGT 2020 efficiency task submission](https://github.com/OpenNMT/CTranslate2/tree/master/examples/wngt2020).

* Set the compute type to "auto" to automatically select the fastest execution path on the current system
* Reduce the beam size to the minimum value that meets your quality requirement
* When using a beam size of 1, keep `return_scores` disabled if you are not using prediction scores: the final softmax layer can be skipped
* Set `max_batch_size` and pass a larger batch to `*_batch` methods: the input sentences will be sorted by length and split by chunk of `max_batch_size` elements for improved efficiency
* Prefer the "tokens" `batch_type` to make the total number of elements in a batch more constant
* Consider using {ref}`translation:dynamic vocabulary reduction` for translation

**On CPU**

* Use an Intel CPU supporting AVX512
* If you are processing a large volume of data, prefer increasing `inter_threads` over `intra_threads` and use stream methods (methods whose name ends with `_file` or `_iterable`)
* Avoid the total number of threads `inter_threads * intra_threads` to be larger than the number of physical cores
* For single core execution on Intel CPUs, consider enabling packed GEMM (set the environment variable `CT2_USE_EXPERIMENTAL_PACKED_GEMM=1`)

**On GPU**

* Use a larger batch size
* Use a NVIDIA GPU with Tensor Cores (Compute Capability >= 7.0)
* Pass multiple GPU IDs to `device_index` to execute on multiple GPUs
