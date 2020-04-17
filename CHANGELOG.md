## [Unreleased]

### New features

### Fixes and improvements

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
