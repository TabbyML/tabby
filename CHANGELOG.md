## [Unreleased]

### New features

### Fixes and improvements

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
