## [Unreleased]

### New features

### Fixes and improvements

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
