# Performance tips

Below are some general recommendations to further improve performance.

## CPU

* Use int8 quantization
* Use an Intel CPU supporting AVX512
* If you are processing a large volume of data, prefer increasing `inter_threads` over `intra_threads` and use stream methods (methods whose name ends with `_file` or `_iterable`)
* Avoid the total number of threads `inter_threads * intra_threads` to be larger than the number of physical cores
* For single core execution on Intel CPUs, consider enabling packed GEMM (set the environment variable `CT2_USE_EXPERIMENTAL_PACKED_GEMM=1`)

## GPU

* Use a larger batch size whenever possible
* Use a NVIDIA GPU with Tensor Cores (Compute Capability >= 7.0)
* Pass multiple GPU IDs to `device_index` to execute on multiple GPUs

## Translator

* The default beam size for translation is 2, but consider setting `beam_size=1` to improve performance
* When using a beam size of 1, keep `return_scores` disabled if you are not using prediction scores: the final softmax layer can be skipped
* Set `max_batch_size` and pass a larger batch to `*_batch` methods: the input sentences will be sorted by length and split by chunk of `max_batch_size` elements for improved efficiency
* Prefer the "tokens" `batch_type` to make the total number of elements in a batch more constant
* Consider using {ref}`translation:dynamic vocabulary reduction` for translation

```{seealso}
The [WNGT 2020 efficiency task submission](https://github.com/OpenNMT/CTranslate2/tree/master/examples/wngt2020) which applies many of these recommendations to optimize machine translation models.
```

## Generator

* Set `include_prompt_in_result=False` so that the input prompt can be forwarded in the decoder at once
* If the model uses a system prompt, consider passing it to the argument `static_prompt` for it to be cached
* When using a beam size of 1, keep `return_scores` disabled if you are not using prediction scores: the final softmax layer can be skipped
