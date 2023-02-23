## [Unreleased]

### New features

### Fixes and improvements

## [v3.7.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v3.7.0) (2023-02-23)

### Changes

* Rename the "float" compute type to "float32" for clarity. "float" is still accepted for backward compatibility.

### New features

* Add the environment variable `CT2_CUDA_TRUE_FP16_GEMM`. This flag is enabled by default so that FP16 GEMMs are running in full FP16. When disabled, the compute type of FP16 GEMMs is set to FP32, which is what PyTorch and TensorFlow do by default.

### Fixes and improvements

* Improve the numerical precision of Whisper models running in FP16 by setting the FP32 compute type for GEMMs (same behavior as PyTorch)
* Improve support for running the Whisper models with INT16 quantization
* Ensure the Whisper decoding does not continue past `max_length`, which could previously happen when the prompt was longer than `max_length/2`
* Include the EOS score in the score returned by Whisper during greedy search

## [v3.6.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v3.6.0) (2023-02-16)

### New features

* Build the Windows Python wheels with cuDNN to enable GPU execution of Whisper models
* Add the model attribute `Whisper.is_multilingual`

### Fixes and improvements

* Reduce the beam search memory usage by not duplicating the decoder states that are the same in each beam (e.g. the projected memory keys and values)
* Optimize the dot product attention during beam search by moving the query beam dimension to the time dimension
* Fix support of English-only Whisper models
* Include the prefix tokens (if they exist) in the output of `Whisper.generate`
* Log a warning when the model weights are implicitly converted to another type

## [v3.5.1](https://github.com/OpenNMT/CTranslate2/releases/tag/v3.5.1) (2023-02-13)

### Fixes and improvements

* Whisper: fix an incorrect timestamp rule that prevented timestamps to be generated in pairs
* Whisper: ignore the EOS token when applying the length penalty to match the original implementation

## [v3.5.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v3.5.0) (2023-02-10)

### New features

* Add a patience factor for beam search to continue decoding until `beam_size * patience` hypotheses are finished, as described in [Kasai et al. 2022](https://arxiv.org/abs/2204.05424)
* Implement all GELU variants and select them accordingly when converting models:
  * Tanh approximation (already implemented)
  * Sigmoid approximation
  * Reference implementation based on the CDF

### Fixes and improvements

* Fix incorrect outputs of T5 models due to a bug in the CUDA kernel of the RMS normalization
* Raise an error if the Whisper input shape is incorrect
* Optimize the transposition operator used in the multi-head attention when running on GPU
* Remove the upper limit in `python_requires` to facilitate the package installation with tools like Poetry and PDM

## [v3.4.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v3.4.0) (2023-02-03)

### Fixes and improvements

* Fix incorrect vocabulary in M2M100 models after conversion with `transformers>=4.24`
* Fix incorrect model outputs when executing with very large batch sizes on GPU
* Fix memory error in biased decoding: the vector of divergence was read and updated past its length
* Allow setting `prefix_bias_beta` > 0 with `beam_size` == 1
* Prevent timestamps from decreasing during Whisper generation
* Make some error messages more helpful when implementing a custom converter

## [v3.3.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v3.3.0) (2023-01-02)

### New features

* Support T5 models, including the variants T5v1.1 and mT5
* Support loading the model files from memory:
  * Python: see the `files` argument in the constructor of classes loading models
  * C++: see the `models::ModelMemoryReader` class

### Fixes and improvements

* Improve the quantization accuracy of OPT models by applying the [SmoothQuant](https://github.com/mit-han-lab/smoothquant) technique during conversion (pre-computed activation scales should be passed to the converter option `--activation_scales`)
* Fix conversion of BART-like models from HuggingFace that are using a different number of encoder and decoder layers
* Fix compilation when no BLAS CPU backend is selected
* Remove no longer relevant CMake warning when the project is compiled without oneDNN
* Update oneDNN to 3.0
* Update oneMKL to 2023.0

## [v3.2.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v3.2.0) (2022-12-12)

### New features

* Add decoding option `suppress_sequences` to prevent specific sequences of tokens from being generated
* Add decoding option `end_token` to stop the decoding on a different token than the model EOS token
* Allow returning multiple random hypotheses from greedy search + random sampling when setting `num_hypotheses` > 1

### Fixes and improvements

* Improve support for batch generation with the Whisper model:
  * Improve performance of batch generation with a context (we only require the prompts to have the same length, which is easily done by adapting the number of previous text tokens)
  * Support batch mode for option `return_no_speech_prob`
  * Support cases where some prompts in the batch have the token `<|notimestamps|>` but not others
* Enable the Conv1D layer in more Python wheels:
  * macOS x64 (using oneDNN)
  * macOS ARM64 (using a custom implementation)
  * Linux AArch64 (using a custom implementation)
* Update the OpenNMT-py converter to support the latest checkpoint structure
* Generalize the `TransformerSpec` constructor to accept arbitrary encoder and decoder specifications
* Remove the global compilation flag `-ffast-math` which introduces unwanted side effects and enable it only for the layer norm CPU kernel where it is actually useful
* Fix CMake error on Windows when setting `-DOPENMP_RUNTIME=COMP`

## [v3.1.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v3.1.0) (2022-11-29)

### Changes

* The input prompt is no longer included in the result of `Whisper.generate` as it is usually not useful in a transcription loop
* The default beam size in `Whisper.generate` is updated from 1 to 5 to match the default value in [openai/whisper](https://github.com/openai/whisper)
* Generation options `min_length` and `no_repeat_ngram_size` now penalize the logits instead of the log probs which may change some scores
* Raise a deprecation warning when reading the `TranslationResult` object as a list of dictionaries

### New features

* Allow configuring the C++ logs from Python with the function `ctranslate2.set_log_level`
* Implement the timestamp decoding rules when the Whisper prompt does not include the token `<|notimestamps|>`
* Add option `return_no_speech_prob` to the method `Whisper.generate` for the result to include the probability of the no speech token

### Fixes and improvements

* Improve performance of the Whisper model when generating with a context
* Fix timestamp tokens in the Whisper vocabulary to use the correct format (`<|X.XX|>`)
* Fix AVX and NEON log functions to return -inf on log(0) instead of NaN
* When info logs are enabled, log the system configuration only when the first model is loaded and not immediately when the library is loaded
* Define a `LogitsProcessor` abstract class to apply arbitrary updates to the logits during decoding
* Update oneDNN to 2.7.2

## [v3.0.2](https://github.com/OpenNMT/CTranslate2/releases/tag/v3.0.2) (2022-11-14)

### Fixes and improvements

* Whisper: fix `generate` arguments that were not correctly passed to the model

## [v3.0.1](https://github.com/OpenNMT/CTranslate2/releases/tag/v3.0.1) (2022-11-10)

### Fixes and improvements

* Whisper: do not implicitly add `<|startoftranscript|>` in `generate` since it is not always the first token

## [v3.0.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v3.0.0) (2022-11-07)

This major version integrates the Whisper speech recognition model published by OpenAI. It also introduces some breaking changes to remove deprecated usages and simplify some modules.

### Breaking changes

#### General

* Remove option `normalize_scores`: the scores are now always divided by `pow(length, length_penalty)` with `length_penalty` defaulting to 1
* Remove option `allow_early_exit`: the beam search now exits early only when no penalties are used

#### Python

* Rename some classes:
  * `OpenNMTTFConverterV2` -> `OpenNMTTFConverter`
  * `TranslationStats` -> `ExecutionStats`
* Remove compatibility for reading `ScoringResult` as a list of scores: the scores can be accessed with the attribute `log_probs`
* Remove compatibility for reading `ExecutionStats` as a tuple
* Remove support for deprecated Python version 3.6

#### CLI

* Rename the client executable `translate` to a more specific name `ct2-translator`

#### C++

* Rename or remove some classes and methods:
  * `TranslationStats` -> `ExecutionStats`
  * `GeneratorPool` -> `Generator`
  * `TranslatorPool` -> `Translator`
  * `TranslatorPool::consume_*` -> `Translator::translate_*`
  * `TranslatorPool::consume_stream` -> removed
  * `TranslatorPool::score_stream` -> removed
* Remove support for building with CUDA 10

### New features

* Integrate the Whisper speech recognition model published by OpenAI
* Support conversion of models trained with OpenNMT-py V3
* Add method `Generator.forward_batch` to get the full model output for a batch of sequences
* Add Python class `StorageView` to expose C++ methods taking or returning N-dimensional arrays: the class implements the array interface for interoperability with Numpy and PyTorch
* Add a new configuration file `config.json` in the model directory that contains non structual model parameters (e.g. related to the input, the vocabulary, etc.)
* Implement the Conv1D layer and operator on CPU and GPU (using oneDNN and cuDNN respectively)
* [C++] Allow registration of external models with `models::ModelFactory`

### Fixes and improvements

* Fix conversion of models that use biases only for some QKV projections but not for all
* Fuse masking of the output log probs by aggregating disabled tokens from all related options: `disable_unk`, `min_length`, `no_repeat_ngram_size`, etc.
* Reduce the layer norm epsilon value on GPU to 1e-5 to match the default value in PyTorch
* Move some Transformer model attributes under the encoder/decoder scopes to simplify loading
* Redesign the `ReplicaPool` base class to simplify adding new classes with multiple model workers
* Compile the library with C++17
* Update oneDNN to 2.7.1
* Update oneMKL to 2022.2
* Update pybind11 to 2.10.1
* Update cibuildwheel to 2.11.2

## [v2.24.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.24.0) (2022-10-03)

### Changes

* The Linux binaries now use the GNU OpenMP runtime instead of Intel OpenMP to workaround an initialization error on systems without `/dev/shm`

### Fixes and improvements

* Fix a memory error when running random sampling on GPU
* Optimize the model loading on multiple GPUs by copying the finalized model weights instead of reading the model from disk multiple times
* In the methods `Translator.translate_iterable` and `Translator.score_iterable`, raise an error if the input iterables don't have the same length
* Fix some compilation warnings

## [v2.23.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.23.0) (2022-09-16)

### New features

* Build wheels for Python 3.11

### Fixes and improvements

* In beam search, get more candidates from the model output and replace finished hypotheses by these additional candidates
* Fix possibly incorrect attention vectors returned from the beam search
* Fix coverage penalty that was actually not applied
* Fix crash when the beam size is larger than the vocabulary size
* Add missing compilation flag `-fvisibility=hidden` when building the Python module
* Update oneDNN to 2.6.2
* Update OpenBLAS to 0.3.21

## [v2.22.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.22.0) (2022-09-02)

### Changes

* `score_batch` methods now return a list of `ScoringResult` instances instead of plain lists of probabilities. In most cases you should not need to update your code: the result object implements the methods `__len__`, `__iter__`, and `__getitem__` so that it can still be used as a list.

### New features

* Add methods to efficiently process long iterables:
  * `Translator.translate_iterable`
  * `Translator.score_iterable`
  * `Generator.generate_iterable`
  * `Generator.score_iterable`
* Add decoding option `min_alternative_expansion_prob` to filter out unlikely alternatives in `return_alternatives` mode
* Return `ScoringResult` instances from `score_batch` to include additional outputs. The current attributes are:
  * `tokens`: the list of tokens that were actually scored (including special tokens)
  * `log_probs`: the log probability of each scored token
* Support running `score_batch` asynchronously by setting the `asynchronous` flag

### Fixes and improvements

* Fix possibly incorrect results when using `disable_unk` or `use_vmap` with one of the following options:
  * `min_decoding_length`
  * `no_repeat_ngram_size`
  * `prefix_bias_beta`
  * `repetition_penalty`
* Also pad the output layer during scoring to enable Tensor Cores
* Improve the correctness of the model output probabilities when the output layer is padded
* Skip translation when the NLLB input is empty (i.e. when the input only contains EOS and the language token)

## [v2.21.1](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.21.1) (2022-07-29)

### Fixes and improvements

* Fix conversion of NLLB models when `tokenizer_class` is missing from the configuration

## [v2.21.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.21.0) (2022-07-27)

### New features

* Support NLLB multilingual models via the Transformers converter
* Support Pegasus summarization models via the Transformers converter

### Fixes and improvements

* Do not stop decoding when the EOS token is coming from the user input: this is required by some text generation models like `microsoft/DialoGPT` where EOS is used as a separator
* Fix conversion error for language models trained with OpenNMT-py
* Fix conversion of models that are not using bias terms in the multi-head attention
* Fix data type error when enabling the translation options `return_alternatives` and `return_attention` with a `float16` model
* Improve CPU performance of language models quantized to `int8`
* Implement a new vectorized GELU operator on CPU
* Raise a more explicit error when trying to convert a unsupported Fairseq model
* Update pybind11 to 2.10.0

## [v2.20.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.20.0) (2022-07-06)

### New features

* Generation option `no_repeat_ngram_size` to prevent the repetitions of N-grams with a minimum size

### Fixes and improvements

* Fix conversion of OpenNMT-tf models that use static position embeddings
* Fix a segmentation fault in `return_alternatives` mode when the target prefix is longer than `max_decoding_length`
* Fix inconsistent state of asynchronous results in Python when a runtime exception is raised
* Remove `<pad>` token when converting MarianMT models from Transformers: this token is only used to start the decoder from a zero embedding, but it is not included in the original Marian model
* Optimize CPU kernels with vectorized reduction of accumulated values
* Do not modify the configuration passed to `OpenNMTTFConverterV2.from_config`
* Improve Python classes documentation by listing members at the top

## [v2.19.1](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.19.1) (2022-06-23)

### Fixes and improvements

* Fix missing final bias in some MarianMT models converted from Transformers
* Fix missing final layer normalization in OPT models converted from Transformers
* Fix error when converting OpenNMT-tf V1 checkpoints with the new OpenNMT-tf converter
* Reduce model conversion memory usage when the loaded weights are in FP16 and the model is converted with quantization
* Add missing C++ type `ctranslate2::float16_t` in the public headers that is required to use some functions
* Fix some Python typing annotations

## [v2.19.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.19.0) (2022-06-08)

### New features

* Support conversion of decoder-only Transformer models trained with OpenNMT-tf

### Fixes and improvements

* Fix conversion error for Transformers' model `facebook/bart-large-cnn`
* Fix crash when scoring empty sequences
* Apply `max_input_length` after all special tokens have been added to the input
* Clear the GPU memory cache when no new batches are immediately available for execution
* Improve functions signature in the generated Python API documentation
* Update oneDNN to 2.6
* Update spdlog to 1.10.0
* Update OpenBLAS to 0.3.20

## [v2.18.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.18.0) (2022-05-23)

### New features

* Support Meta's OPT models via the Transformers converter
* Extend the Fairseq converter to support `transformer_lm` models

### Fixes and improvements

* Fix conversion error for Marian's pre-norm Transformer models
* Fix conversion error for Transformers' MarianMT models that are missing some configuration fields
* Improve conversion speed of Marian models (optimize the generation of the sinusoidal position encodings)

## [v2.17.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.17.0) (2022-05-09)

### New features

* Add a converter for Hugging Face's [Transformers](https://github.com/huggingface/transformers). The following models are currently supported:
  * BART
  * M2M100
  * MarianMT
  * MBART
  * OpenAI GPT2
* Revisit the OpenNMT-tf converter to better support custom models and configurations:
  * Extend the conversion script to accept the training configuration
  * Add a new converter class `ctranslate2.converters.OpenNMTTFConverterV2`
* Move all documentation and guides to the [website](https://opennmt.net/CTranslate2) to improve navigation and clarity

### Fixes and improvements

* In text generation, include the start token in the output if it is not the BOS token

## [v2.16.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.16.0) (2022-04-28)

### New features

* Initial support of language models:
  * Add a high-level class `ctranslate2.Generator` to generate text with language models
  * Add a converter for OpenAI GPT-2 models
  * Update the OpenNMT-py converter to support `transformer_lm` decoders
* Build ARM64 wheels for macOS
* Allow loading custom Fairseq extensions and architectures during conversion with the option `--user_dir`
* Enable conversion of the Fairseq architectures `multilingual_transformer` and `multilingual_transformer_iwslt_de_en`
* Implement random sampling in beam search using the Gumbel-max trick
* Generate and publish the Python API reference to https://opennmt.net/CTranslate2

### Fixes and improvements

* Fix model loading on a GPU with index > 0
* Fix memory error when running random sampling on GPU with certain batch sizes
* Fix incorrect tokens order in some converted Marian vocabularies
* Properly count the number of layers before building the encoder/decoder instead of relying on runtime exceptions

## [v2.15.1](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.15.1) (2022-04-04)

### Fixes and improvements

* Fix missing deactivation of OpenMP threading in GPU execution (regression introduced in version 2.15.0)

## [v2.15.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.15.0) (2022-04-04)

### New features

* Expose translator option `max_queued_batches` to configure the maximum number of queued batches (when the queue is full, future requests will block until a free slot is available)
* Allow converters to customize the vocabulary special tokens `<unk>`, `<s>`, and `</s>`

### Fixes and improvements

* Fix compatibility of models converted on Windows with other platforms by saving the vocabulary files with the newline character "\n" instead of "\r\n"
* Clarify conversion error when no TensorFlow checkpoints are found in the configured model directory
* Enable fused QKV transposition by switching the heads and time dimensions before the QKV split
* Cache the prepared source lengths mask in the Transformer decoder state and reuse it in the next decoding steps
* Pad the output layer to enable Tensor Cores only once instead of updating the layer on each batch
* Vectorize copy in Concat and Split ops on GPU
* Factorize all OpenMP parallel for loops to call the `parallel_for` function
* Compile CUDA kernels for deprecated Compute Capabilities that are not yet dropped by CUDA:
  * CUDA 11: 3.5 and 5.0
  * CUDA 10: 3.0

## [v2.14.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.14.0) (2022-03-16)

### New features

* Include BART and MBART in the list of supported Fairseq architectures
* Add Fairseq converter option `--no_default_special_tokens` to require all special tokens to be set by the user during inference, including the decoder start tokens (for example, this is required by MBART-25 to properly set the language tokens)

### Fixes and improvements

* Fix conversion of Post-Norm Transformers trained with OpenNMT-tf
* Fix scoring with Fairseq models that used an incorrect decoder start token (Fairseq uses `</s>` as the decoder start token, not `<s>`)
* Fix scoring result to include the end of sentence token
* Ignore OpenNMT-py options `--alignment_layer` and `--alignment_heads` for models that are not trained with alignments
* Enable batch encoding in `return_alternatives` translation mode (the decoding still runs sequentially)
* Make enumerations `ctranslate2.specs.Activation` and `ctranslate2.specs.EmbeddingsMerge` public since they could be used to configure the Transformer specification
* Update oneDNN to 2.5.3
* Update cpu_features to 0.7.0
* Update cxxopts to 3.0.0
* Update spdlog to 1.9.2

## [v2.13.1](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.13.1) (2022-03-02)

### Fixes and improvements

* Fix conversion error for old OpenNMT-py models that do not have the option `self_attn_type`

## [v2.13.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.13.0) (2022-02-28)

### New features

* Add converter for [Marian](https://github.com/marian-nmt/marian) and support the collection of [OPUS-MT pretrained models](https://github.com/Helsinki-NLP/Opus-MT-train/tree/master/models)
* Support models applying a layer normalization after the embedding layer (cf. option `--layernorm-embedding` in Fairseq)
* Support models using the Swish (a.k.a SiLU) activation function
* Support models using custom decoder start tokens, which can be passed in the target prefix

### Fixes and improvements

* Remove unexcepted call to a CUDA function in CPU execution when unloading models
* Add option groups in the translation client help output
* Use new `thrust::cuda::par_nosync` execution policy when calling Thrust functions
* Update Thrust to 1.16.0
* Update pybind11 to 2.9.1

## [v2.12.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.12.0) (2022-02-01)

### New features

* Support models using additional source features (a.k.a. factors)

### Fixes and improvements

* Fix compilation with CUDA < 11.2
* Fix incorrect revision number reported in the error message for unsupported model revisions
* Improve quantization correctness by rounding the value instead of truncating (this change will only apply to newly converted models)
* Improve default value of `intra_threads` when the system has less than 4 logical cores
* Update oneDNN to 2.5.2

## [v2.11.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.11.0) (2022-01-11)

### Changes

* With CUDA >= 11.2, the environment variable `CT2_CUDA_ALLOCATOR` now defaults to `cuda_malloc_async` which should improve performance on GPU.

### New features

* Build Python wheels for AArch64 Linux

### Fixes and improvements

* Improve performance of Gather CUDA kernel by using vectorized copy
* Update Intel oneAPI to 2022.1
* Update oneDNN to 2.5.1
* Log some additional information with `CT2_VERBOSE` >= 1:
  * Location and compute type of loaded models
  * Version of the dynamically loaded cuBLAS library
  * Selected CUDA memory allocator

## [v2.10.1](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.10.1) (2021-12-15)

### Fixes and improvements

* Fix stuck execution when loading a model on a second GPU
* Fix numerical error in INT8 quantization on macOS

## [v2.10.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.10.0) (2021-12-13)

### Changes

* `inter_threads` now also applies to GPU translation, where each translation thread is using a different CUDA stream to allow some parts of the GPU execution to overlap

### New features

* Add option `disable_unk` to disable the generation of unknown tokens
* Add function `set_random_seed` to fix the seed in random sampling
* [C++] Add constructors in `Translator` and `TranslatorPool` classes with `ModelReader` parameter

### Fixes and improvements

* Fix incorrect output from the Multinomial op when running on GPU with a small batch size
* Fix Thrust and CUB headers that were included from the CUDA installation instead of the submodule
* Fix static library compilation with the default build options (`cmake -DBUILD_SHARED_LIBS=OFF`)
* Compile the Docker image and the Linux Python wheels with SSE 4.1 (vectorized kernels are still compiled for AVX and AVX2 with automatic dispatch, but other source files are now compiled with SSE 4.1)
* Enable `/fp:fast` for MSVC to mirror `-ffast-math` that is enabled for GCC and Clang
* Statically link against oneDNN to reduce the size of published binaries:
  * Linux Python wheels: 43MB -> 17MB
  * Windows Python wheels: 41MB -> 11MB
  * Docker image: 733MB -> 600MB

## [v2.9.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.9.0) (2021-12-01)

### New features

* Add GPU support to the Windows Python wheels
* Support OpenNMT-py and Fairseq options `--alignment_layer` and `--alignment_heads` which specify how the multi-head attention is reduced and returned by the Transformer decoder
* Support dynamic loading of CUDA libraries on Windows

### Fixes and improvements

* Fix division by zero when normalizing the score of an empty target
* Fix error that was not raised when the input length is greater than the number of position encodings
* Improve performance of random sampling on GPU for large values of `sampling_topk` or when sampling over the full vocabulary
* Include `transformer_align` and `transformer_wmt_en_de_big_align` in the list of supported Fairseq architectures
* Add a CUDA kernel to prepare the length mask to avoid moving back to the CPU

## [v2.8.1](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.8.1) (2021-11-17)

### Fixes and improvements

* Fix dtype error when reading float16 scores in greedy search
* Fix usage of MSVC linker option `/nodefaultlib` that was not correctly passed to the linker

## [v2.8.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.8.0) (2021-11-15)

### Changes

* The Linux Python wheels now use Intel OpenMP instead of GNU OpenMP for consistency with other published binaries

### New features

* Build Python wheels for Windows

### Fixes and improvements

* Fix segmentation fault when calling `Translator.unload_model` while an asynchronous translation is running
* Fix implementation of repetition penalty that should be applied to all previously generated tokens and not just the tokens of the last step
* Fix missing application of repetition penalty in greedy search
* Fix incorrect token index when using a target prefix and a vocabulary mapping file
* Set the OpenMP flag when compiling on Windows with `-DOPENMP_RUNTIME=INTEL` or `-DOPENMP_RUNTIME=COMP`

## [v2.7.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.7.0) (2021-11-03)

### Changes

* Inputs are now truncated after 1024 tokens by default (see translation option `max_input_length`)

### New features

* Add translation option `max_input_length` to limit the model input length
* Add translation option `repetition_penalty` to apply an exponential penalty on repeated sequences
* Add scoring option `with_tokens_score` to also output token-level scores when scoring a file

### Fixes and improvements

* Adapt the length penalty formula when using `normalize_scores` to match other implementations: the scores are divided by `pow(length, length_penalty)`
* Implement `LayerNorm` with a single CUDA kernel instead of 2
* Simplify the beam search implementation

## [v2.6.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.6.0) (2021-10-15)

### New features

* Build wheels for Python 3.10
* Accept passing the vocabulary as a `opennmt.data.Vocab` object or a list of tokens in the OpenNMT-tf converter

### Fixes and improvements

* Fix segmentation fault in greedy search when `normalize_scores` is enabled but not `return_scores`
* Fix segmentation fault when `min_decoding_length` and `max_decoding_length` are both set to 0
* Fix segmentation fault when option `sampling_topk` is larger than the vocabulary size
* Fix incorrect score normalization in greedy search when `max_decoding_length` is reached
* Fix incorrect score normalization in the `return_alternatives` translation mode
* Improve error checking when reading the binary model file
* Apply `LogSoftMax` in-place during decoding and scoring

## [v2.5.1](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.5.1) (2021-10-04)

### Fixes and improvements

* Fix logic error in the in-place implementation of the `Gather` op that could lead to incorrect beam search outputs

## [v2.5.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.5.0) (2021-10-01)

### New features

* Add an 8-bit GEMM backend on AArch64 using [Ruy](https://github.com/google/ruy)

### Fixes and improvements

* Skip unnecessary transpositions of the projected decoder queries in the multi-head attention
* Use 32-bit indexing in all CUDA kernels to slightly improve performance
* Let the compiler auto-vectorize the `LayerNorm` CPU kernel
* Update Intel oneAPI to 2021.4

## [v2.4.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.4.0) (2021-09-10)

### New features

* [Python] Support asynchronous translation: `translate_batch` can return future-like objects with argument `asynchronous=True`
* [Python] `translate_batch` now returns a list of `TranslationResult` objects instead of a list of dictionaries (this object can also be indexed as a list of dictionaries for backward compatibility)
* Add options `--source_lang` and `--target_lang` to the Fairseq converter for models that do not include these information

### Fixes and improvements

* Fix Fairseq model conversion when the model options are stored in `model["cfg"]["model"]`
* Compile the CPU INT8 quantization kernel with FMA instructions
* Enable packing of the last linear weight when not using dynamic vocabulary reduction
* Replace the generic `Tile` implementation by dedicated CPU and CUDA kernels
* [Python] Implement `__repr__` method for `TranslationStats` objects
* [Python] Update pybind11 to 2.7.1

## [v2.3.2](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.3.2) (2021-08-05)

### Fixes and improvements

* Fix GPU execution that gets stuck when applying the GELU activation

## [v2.3.1](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.3.1) (2021-07-28)

### Fixes and improvements

* Fix compilation with CUDA < 10.2

## [v2.3.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.3.0) (2021-07-26)

### New features

* Add compute type `int8_float16` for mixed INT8 and FP16 computation on GPU (requires Compute Capability >= 7.0)
* Add methods `Translator.score_batch` and `Translator.score_file` to score existing translations

### Fixes and improvements

* Relax the GPU driver requirement for running the Docker image to >= 450.80.02 (same as the published Python package)

## [v2.2.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.2.0) (2021-07-06)

### New features

* Add Python utility functions to query the system capabilities:
  * `ctranslate2.get_cuda_device_count`
  * `ctranslate2.get_supported_compute_types`
* Add option `fixed_dictionary` in the Fairseq converter to support multilingual models
* Extend environment variable `CT2_VERBOSE` to configure more log levels (see README)

### Fixes and improvements

* Fuse activation with bias addition on GPU for a small performance increase
* Make the GELU activation compatible with FP16 execution
* Improve the log format using the [spdlog](https://github.com/gabime/spdlog) library
* Improve the accuracy of the profiling results on GPU
* Update Intel oneAPI to 2021.3

## [v2.1.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.1.0) (2021-06-14)

### New features

* Support conversion of Transformer models trained with [Fairseq](https://github.com/pytorch/fairseq/) (see script `ct2-fairseq-converter`)
* Support conversion of models using GELU activations
* Add translation option `normalize_scores` to return scores normalized by the hypotheses length: enabling this option can improve the beam search output for some models
* Add translation option `allow_early_exit` to toggle the beam search early exit optimization: disabling this option has a small negative impact on performance, but it can improve the beam search output when using penalties or normalized scores
* [C++] Add class `BufferedTranslationWrapper` to buffer and batch independent inputs to the same model

### Fixes and improvements

* Read value of environment variable `OMP_NUM_THREADS` when `intra_threads` is not set
* Improve file translation performance by enabling local sorting by default
* [Python] Improve error message when converting unsupported models and list all options that are unuspported
* [Python] Return statistics of `Translator.translate_file` as an object with named properties
* [C++] Fix compilation of method `TranlatorPool::consume_raw_text_file` that takes streams as inputs

## [v2.0.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v2.0.0) (2021-06-03)

This major version introduces some breaking changes to simplify model conversion, improve the consistency of user options, and update the Python package to CUDA 11.x. It also comes with internal improvements to facilitate future changes.

### Breaking changes

#### General

* Disable `return_scores` by default as most applications do not use translation scores
* Replace all Docker images by a single one: `<version>-ubuntu20.04-cuda11.2`
* Replace CMake option `LIB_ONLY` by `BUILD_CLI`
* Require CMake version >= 3.15 for GPU compilation

#### Python

* For GPU execution, the Linux Python wheels published on PyPI now require CUDA 11.x to be installed on the system. The CUDA dependencies (e.g. cuBLAS) are no longer included in the package and are loaded dynamically.
* Remove support for converting the TensorFlow SavedModel format (checkpoints should be converted instead)
* Remove the `model_spec` option for converters that can automatically detect it from the checkpoints
* Force translation options to be set with keyword arguments only (see the API reference)
* Rename tokenization callables arguments in `translate_file` for clarity:
  * `tokenize_fn` to `source_tokenize_fn`
  * `detokenize_fn` to `target_detokenize_fn`

#### CLI

* Rename length contraints options for consistency with other APIs:
  * `max_sent_length` to `max_decoding_length`
  * `min_sent_length` to `min_decoding_length`

#### C++

* Move the `max_batch_size` and `batch_type` options from the `TranslationOptions` structure to the translation methods of `TranslatorPool`
* Simplify the `TranslationResult` structure with public attributes instead of methods
* Asynchronous translation API now returns one future per example instead of a single future for the batch

### New features

* Add translation option `prefix_bias_beta` to bias the decoding towards the target prefix (see [Arivazhagan et al. 2020](https://arxiv.org/abs/1912.03393))
* Automatically detect the model specification when converting OpenNMT-py models
* Support conversion and execution of Post-Norm Transformers
* Add an experimental asynchronous memory allocator for CUDA 11.2 and above (can be enabled with the environment variable `CT2_CUDA_ALLOCATOR=cuda_malloc_async`)
* Expose the Python package version in `ctranslate2.__version__`

### Fixes and improvements

* Fix silent activation of `replace_unknowns` when enabling `return_attention`
* Improve support for the NVIDIA Ampere architecture in prebuilt binaries
* Reduce the size of the Python wheels published on PyPI
* Define a custom CUDA kernel for the GEMM output dequantization instead of a Thrust-based implementation
* Update Thrust to 1.12.0

## [v1.20.1](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.20.1) (2021-04-29)

### Fixes and improvements

* Do not return scores for empty outputs when `return_scores` is disabled
* Do not include google/cpu\_features library in CTranslate2 installation

## [v1.20.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.20.0) (2021-04-20)

### Changes

* Drop Python 3.5 support
* Docker image tags suffixed with `-gpu` are no longer updated to prefer tags with an explicit CUDA version

### Fixes and improvements

* Fix int8 quantization for rows that only contains zeros
* Fix type error when running the CUDA code path of the Multinomial operator
* Add EOS score to the greedy search final score for consistency with the beam search output
* Use third party library [google/cpu\_features](https://github.com/google/cpu_features) to resolve CPU features at runtime
* Small optimizations when manipulating tensor shapes and indices
* Internal refactoring of Transformer layers

## [v1.19.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.19.0) (2021-03-31)

### Changes

* Rename CMake option `WITH_TESTS` to `BUILD_TESTS`

### New features

* Add "auto" compute type to automatically select the fastest compute type on the current system

### Fixes and improvements

* [Python] Clear memory allocator cache when calling `unload_model`
* [Python] Make methods `unload_model` and `load_model` thread safe
* Fix conversion of TensorFlow SavedModel with shared embeddings
* Update Intel oneAPI to 2021.2
* Compile core library with C++14 standard

## [v1.18.3](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.18.3) (2021-03-02)

### Fixes and improvements

* Use Intel OpenMP instead of GNU OpenMP in the Docker images as a workaround for issue #409.

## [v1.18.2](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.18.2) (2021-02-23)

### Fixes and improvements

* Fix crash when enabling coverage penalty in GPU translation
* Fix incorrect value of AVX2 flag in `CT2_VERBOSE` output

## [v1.18.1](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.18.1) (2021-02-01)

### Fixes and improvements

* Fix conversion of models setting the attributes `with_source_bos` or `with_source_eos`

## [v1.18.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.18.0) (2021-01-28)

### Changes

* Some options default value in the `translate` client have been changed to match the Python API:
  * `batch_size` = 32 (instead of 30)
  * `beam_size` = 2 (instead of 5)
  * `intra_threads` = 4 (instead of 0)

### New features

* Support multi-GPU translation: `device_index` argument can now be set to a list of GPU IDs (see [example](https://github.com/OpenNMT/CTranslate2/blob/master/docs/python.md#note-on-parallel-translations))

### Fixes and improvements

* Improve performance when using multiple GPU translators concurrently in the same process
* [Python] Do nothing when calling `unload_model(to_cpu=True)` on CPU translators
* [Python] Set a default value for `max_batch_size` argument in method `Translator.translate_file`
* Disable `CT2_TRANSLATORS_CORE_OFFSET` in OpenMP builds as setting thread affinity does not work when OpenMP is enabled

## [v1.17.1](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.17.1) (2021-01-15)

### Fixes and improvements

* Fix Python wheel loading error on macOS

## [v1.17.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.17.0) (2021-01-11)

### Changes

* Linux Python wheels are now compiled under `manylinux2014` and require `pip` version >= 19.3

### New features

* Publish Python wheels for macOS (CPU only)
* Support compilation for ARM 64-bit architecture and add NEON vectorization
* Add new optional GEMM backends: [Apple Accelerate](https://developer.apple.com/documentation/accelerate) and [OpenBLAS](https://www.openblas.net/)
* Add `replace_unknowns` translation option to replace unknown target tokens by source tokens with the highest attention
* Add flags in the model specification to declare that BOS and/or EOS tokens should be added to the source sequences

### Fixes and improvements

* Fix segmentation fault when the model is converted with a wrong vocabulary and predicts an out-of-vocabulary index
* Fix result of vectorized array reduction when the array length is not a multiple of the SIMD registers width
* Fix exit code when running `cli/translate -h`
* Improve performance of vectorized vector math by inlining calls to intrinsics functions
* Improve accuracy of LogSoftMax CUDA implementation
* Improve error message when `--model` option is not set in `cli/translate`
* Update oneMKL to 2020.1 in published binaries
* Update oneDNN to 2.0 in published binaries
* Update default search paths to support compilation with oneMKL and oneDNN installed from the oneAPI toolkit

## [v1.16.2](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.16.2) (2020-11-27)

### Fixes and improvements

* Fix cuBLAS version included in the Python wheels published to PyPI. The included library was targetting CUDA 10.2 instead of CUDA 10.1.
* Re-add Python 3.5 wheels on PyPI to give users more time to transition

## [v1.16.1](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.16.1) (2020-11-23)

### Fixes and improvements

* Fuse dequantization and bias addition on GPU for improved INT8 performance
* Improve performance of masked softmax on GPU
* Fix error when building the CentOS 7 GPU Docker image
* The previous version listed "Pad size of INT8 matrices to a multiple of 16 when the GPU has INT8 Tensor Cores". However, the padding was not applied due to a bug and fixing it degraded the performance, so this behavior is not implemented for now.

## [v1.16.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.16.0) (2020-11-18)

### Changes

* Drop support for Python 2.7 and 3.5

### New features

* Add Docker images using CUDA 11.0

### Fixes and improvements

* Enable parallel CPU translations from `translate_batch` in Python when setting `inter_threads` > 1 and `max_batch_size` > 0
* Improve GPU performance on Turing architecture when using a Docker image or the Python package
* Pad size of INT8 matrices to a multiple of 16 when the GPU has INT8 Tensor Cores
* Add information about detected GPU devices in `CT2_VERBOSE` output
* Update oneDNN to 1.7
* [Python] Improve type checking for some arguments

## [v1.15.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.15.0) (2020-11-06)

### New features

* [Experimental] The Python package published on PyPI now includes GPU support. The binary is compiled with CUDA 10.1, but all CUDA dependencies are integrated in the package and do not need to be installed on the system. The only requirement should be a working GPU with driver version >= 418.39.

### Fixes and improvements

* Remove the TensorRT dependency to simplify installation and reduce memory usage:
  * Reduce GPU Docker images size by 600MB
  * Reduce memory usage on the GPU and the system by up 1GB
  * Reduce initialization time during the first GPU translation
* Improve TopK performance on GPU for K < 5
* Improve INT8 performance on GPU
* Accept linear layers without bias when converting models
* Update Intel MKL to 2020.4
* [Python] Improve compatibility with Python 3.9

## [v1.14.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.14.0) (2020-10-13)

### New features

* Accept target prefix in file translation APIs

### Fixes and improvements

* Fix CUDA illegal memory access when changing the beam size in the same process
* Fix decoding with target prefix that sometimes did not go beyond the prefix
* Fix Intel MKl search paths on macOS
* Update Intel MKL to 2020.3
* Clarify error message when selecting a CUDA device in CPU-only builds

## [v1.13.2](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.13.2) (2020-08-31)

### Fixes and improvements

* Fix model conversion to `float16` when using the Python converters: weights were duplicated and not correctly converted
* Fix incorrect code logic that could lead to incorrect translation results

## [v1.13.1](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.13.1) (2020-08-06)

### Fixes and improvements

* Fix performance regression when decoding with a large beam size on GPU

## [v1.13.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.13.0) (2020-07-30)

### New features

* Environment variable `CT2_TRANSLATORS_CORE_OFFSET` to pin parallel translators to a range of CPU cores (only for `intra_threads` = 1)
* [Python] Add some properties to the `Translator` object:
  * `device`
  * `device_index`
  * `num_translators`
  * `num_queued_batches`
  * `model_is_loaded`

### Fixes and improvements

* Improve batch performance of target prefix
* Improve performance when the input batch contains sentences with very different lengths
* Improve beam search performance by expanding the batch size only after the first decoding step
* Optimize Transpose op on GPU for the permutation used in multi-head attention
* Remove padding in returned attention vectors
* Update Intel MKL to 2020.2

## [v1.12.1](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.12.1) (2020-07-20)

### Fixes and improvements

* Fix implicit int16 to float16 model conversion on compatible GPUs

## [v1.12.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.12.0) (2020-07-16)

### Changes

* Docker images based on Ubuntu 16.04 are no longer updated

### New features

* Support `float16` data type for model conversion (with `--quantization float16`) and computation (with `--compute_type float16`). FP16 execution can improve performance by up to 50% on NVIDIA GPUs with Compute Capability >= 7.0.
* Add Docker images with newer CUDA versions, which can improve performance in some cases:
  * `latest-ubuntu18-cuda10.0` (same as `latest-ubuntu18-gpu`)
  * `latest-ubuntu18-cuda10.1`
  * `latest-ubuntu18-cuda10.2`
  * `latest-centos7-cuda10.0` (same as `latest-centos7-gpu`)
  * `latest-centos7-cuda10.1`
  * `latest-centos7-cuda10.2`
* Allow setting a computation type per device (e.g. `Translator(..., compute_type={"cuda": "float16", "cpu": "int8"})` with the Python API)
* [C++] Add `ModelReader` interface to customize model loading

### Fixes and improvements

* Optimize Transpose op on CPU for the permutation used in multi-head attention
* Optimize GELU op CPU with Intel MKL
* Fix compilation when targeting an architecture and disabling ISA dispatch (e.g.: `-DCMAKE_CXX_FLAGS="-march=skylake" -DENABLE_CPU_DISPATCH=OFF`)
* Inline some frequently called methods

## [v1.11.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.11.0) (2020-06-29)

### New features

* Add tokenization and detokenization hooks for file translation APIs
* Add alternatives to Intel MKL:
  * Integrate [oneDNN](https://github.com/oneapi-src/oneDNN) for GEMM functions
  * Implement vectorized operators that automatically select the instruction set architecture (ISA) (can be manually controlled with the `CT2_FORCE_CPU_ISA` environment variable)
* When alternatives are available, avoid using Intel MKL on non Intel processors (can be manually controlled with the `CT2_USE_MKL` environment variable)
* Enable a verbose mode with the environment variable `CT2_VERBOSE=1` to help debugging the run configuration (e.g. the detected CPU, whether Intel MKL is being used, etc.)

### Fixes and improvements

* Improve numerical precision of SoftMax and LogSoftMax layers on CPU
* Parallelize INT16 quantization/dequantization and ReLU on CPU
* Add back the translation client in CentOS 7 Docker images

## [v1.10.2](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.10.2) (2020-06-23)

### Fixes and improvements

* [Python] Fix error when calling `unload_model(to_cpu=True)` for models with shared weights
* [Python] Do not ignore errors when importing the compiled translator extension

## [v1.10.1](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.10.1) (2020-05-25)

### Fixes and improvements

* Force `intra_threads` to 1 when running a model on GPU to prevent high CPU load
* Improve handling of decoding length constraints when using a target prefix
* Do not raise an error when setting `use_vmap` but no vocabulary map exists

## [v1.10.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.10.0) (2020-04-17)

### New features

* Coverage penalty as in [Wu et al. 2016](https://arxiv.org/abs/1609.08144) with the option `coverage_penalty`
* Batch size can be expressed in number of tokens with the option `batch_type`
* Translation scores can be disabled with the option `return_scores` (if disabled, the final SoftMax is skipped during greedy decoding)
* Support compilation without TensorRT by setting `-DWITH_TENSORRT=OFF` during CMake configuration (in this case, beam search is no longer supported)
* Experimental integration of [Intel MKL's packed GEMM](https://software.intel.com/en-us/articles/introducing-the-new-packed-apis-for-gemm) which can be enabled by setting the environment variable `CT2_USE_EXPERIMENTAL_PACKED_GEMM=1`

### Fixes and improvements

* Remove direct dependency to cuDNN (still an indirect dependency via TensorRT)
* Static AVX optimization for the ReLU operator
* Remove unnecessary memory initialization when creating temporary buffers
* Dissociate SoftMax and LogSoftMax in profiling report

## [v1.9.1](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.9.1) (2020-04-08)

### Fixes and improvements

* Fix parallel translations when calling `Translator.translate_batch` from multiple Python threads
* Fix crash on invalid `num_hypotheses` value

## [v1.9.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.9.0) (2020-03-24)

### New features

* Return 2 additional statistics from file translation APIs:
  * the number of translated examples
  * the total translation time in milliseconds

### Fixes and improvements

* Fix exceptions that were not catched by the Python wrapper
* Fix an invalid insertion in the variables collection while iterating over it
* Optimize filling operation of float storages
* Internal refactoring of decoding functions to make them reusable for other tasks (e.g. generative language models)

## [v1.8.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.8.0) (2020-03-10)

### New features

* [Python] Add methods `Translator.unload_model` and `Translator.load_model` to manually manage memory
* [Docker] Move all images to Python 3 only
* Expose options that enable an internal sorting by length to increase the translation efficiency:
  * for file translation: `read_batch_size` contiguous examples will be loaded, sorted by length, and batched with size `max_batch_size`
  * for batch translation: if the batch is larger than `max_batch_size`, examples will be sorted by length and batched with size `max_batch_size`

### Fixes and improvements

* Fix another error when releasing a translator that is placed on a GPU that is not GPU 0
* Fix possible memory corruption when creating GPU translators in parallel
* Fix memory that is briefly allocated on GPU 0 when destroying a translator that is placed on another GPU
* Reduce latency of model loading, especially on GPU

## [v1.7.1](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.7.1) (2020-03-03)

### Fixes and improvements

* Revert "Parallelize some low level transformations on CPU" which caused incorrect computation
* Avoid unnecessary TensorFlow runtime initialization when converting checkpoints
* Fix compilation without MKL

## [v1.7.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.7.0) (2020-02-28)

### New features

* Translation option `return_alternatives` to return multiple choices at the first unconstrained decoding position: combined with a target prefix, this could be used to provide alternative words and translations at a specific location in the target
* Support Transformers with different number of encoder/decoder layers
* Allow compilation without OpenMP with `-DOPENMP_RUNTIME=NONE`

### Fixes and improvements

* Fix SavedModel conversion when TensorFlow Addons 0.8 is installed
* Fix error when releasing a translator/model that is placed on a GPU that is not GPU 0
* Fix memory that was allocated on GPU 0 even when the translator/model was placed on another GPU
* Query GPU int8 support on the first model load, and then cache the result for future loads
* Avoid creating an empty model directory on conversion errors
* Parallelize some low level transformations on CPU
* Reduce memory usage when translating large files by limiting the work queue size

## [v1.6.3](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.6.3) (2020-02-24)

### Fixes and improvements

* Fix incorrectness in relative representation computation

## [v1.6.2](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.6.2) (2020-02-21)

### Fixes and improvements

* Fix conversion of models with shared embeddings

## [v1.6.1](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.6.1) (2020-02-11)

### Fixes and improvements

* [Docker] Remove translation client in CentOS 7 images as it can cause compatibility issues with downstream images

## [v1.6.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.6.0) (2020-02-14)

### New features

* Support Transformers with relative position representations (as in [Shaw et al. 2018](https://arxiv.org/abs/1803.02155))
* Accept target prefix in batch request
* Support `return_attention` with prefixed translation

## [v1.5.1](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.5.1) (2020-02-06)

### Fixes and improvements

* Fix INT8 translation on CPU with vocabulary map

## [v1.5.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.5.0) (2020-02-06)

### New features

* [C++] Add `max_batch_size` translation options for single translators

### Fixes and improvements

* Improve INT8 performance on CPU
* Enable INT8 support on default Intel MKL build
* Simplify project dependencies:
  * Replace `boost::program_options` with `cxxopts` for client options
  * Include header-only dependencies as Git submodules (`cxxopts`, `cub`, and `thrust`)
  * Remove MKL-DNN
* Harmonize Python/C++ default values:
  * [Python] Change default beam size from 4 to 2
  * [C++] Load models on the CPU by default

## [v1.4.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.4.0) (2020-01-20)

### New features

* Publish a package on [PyPI](https://pypi.org/project/ctranslate2/) (without GPU support)
* Add method to convert OpenNMT-tf models directly from a dictionary of variables
* Return statistics from Python method `Translator.translate_file`
* Add `set_model` methods to support changing models without creating a new `Translator`
* Add a `contains_model` function to check whether a directory could contain a CTranslate2 model

## [v1.3.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.3.0) (2020-01-14)

### New features

* Support random sampling (see the `sampling_topk` and `sampling_temperature` translation options)
* `CT2_CUDA_CACHING_ALLOCATOR_CONFIG` environment variable to configure the CUDA caching allocator

### Fixes and improvements

* Fix incorrect translations on Windows due to incompatibility between the compiler OpenMP and Intel OpenMP
* Release cuDNN/cuBLAS/TensorRT handles on thread exit when destroying a `TranslatorPool`
* Remove use of `--{start,end}-group` compiler options when compiling on Mac OS
* Update Intel MKL to 2020.0 in Docker images
* Load vocabulary assets for SavedModel exported with OpenNMT-tf 2.5 and above

## [v1.2.3](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.2.3) (2019-12-11)

### Fixes and improvements

* Improve translator robustness on empty batch and inputs
* Speed optimization for `LayerNorm`
* Check vocabulary size when converting OpenNMT-tf models
* Add more samples in the execution profiling output which now supports nested functions

## [v1.2.2](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.2.2) (2019-11-25)

### Fixes and improvements

* Fix `PositionEncoder` internal state that was shared with other instances on the same thread
* Replace Boost.Python by pybind11
* Include a Python source distribution in the Docker images

## [v1.2.1](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.2.1) (2019-11-06)

### Fixes and improvements

* Avoid copying decoder states when possible to improve decoding performance (10% to 20% faster)
* Fix execution profiling on GPU (device was not synchronized before measuring the time)
* Include `Mul` operation in profiling report
* Add a Python 3 wheel in Ubuntu Docker images

## [v1.2.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.2.0) (2019-10-28)

### New features

* Accept Transformer models with custom number of layers and heads
* `--log-profiling` client option to profile ops execution

### Fixes and improvements

* Fix conversion error for models having 2 different weights with the same values
* Fix invalid MKL function override after a refactoring
* Add more information and context to several error messages

## [v1.1.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.1.0) (2019-10-18)

### New features

* New Docker images: `latest-ubuntu16-gpu`, `latest-ubuntu18`, `latest-ubuntu18-gpu`
* Support OpenNMT-tf Transformer models with shared embeddings
* Update to TensorRT 6
* Make OpenMP runtime configurable

### Fixes and improvements

* Reduce the size of models with shared weights on disk and in memory
* Shared words vocabulary is no longer duplicated on disk and in memory
* Improve performance of translation with a vocabulary map on GPU
* Statically link against Intel MKL
* Remove some implementation details from public headers

## [v1.0.1](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.0.1) (2019-10-08)

### Fixes and improvements

* Fix loading of newer OpenNMT-py models
* Promote FP16 to FP32 in model converter scripts
* Improve INT8 performance on CPU and GPU
* Improve performance on GPU by fusing the layer normalization operation `x * gamma + beta`
* Enable INT8 and INT16 computation on all platforms with Intel MKL 2019.5 and above

## [v1.0.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v1.0.0) (2019-09-23)

First stable release.
