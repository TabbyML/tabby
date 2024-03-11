# Contributing

This document provides some information to help you contribute to the CTranslate2.

## Reporting issues

We use GitHub issues for bugs in the code that are **reproducible**. A good bug report should contain every information needed to reproduce it. Before opening a new issue, make sure to:

* **use the GitHub issue search** for existing and fixed bugs;
* **check if the issue has been fixed** in a more recent version;
* **isolate the problem** to give as much context as possible.

If you have questions on how to use the project or have trouble getting started with it, consider using [our forum](https://forum.opennmt.net/) instead and tagging your topic with the *ctranslate2* tag.

## Requesting features

Do you think a feature is missing or would be a great addition to the project? Please open a GitHub issue to describe it.

## Developing code

* If you want to contribute with code but are unsure what to do,
  * search for *TODO* comments in the code: these are small dev tasks that should be addressed at some point.
  * look for GitHub issues marked with the *help wanted* label: these are developments that we find particularly suited for community contributions.
* If you are planning to make a large change to the existing code, consider asking first on [the forum](https://forum.opennmt.net/) to confirm that it is welcome.

### Building the sources

See [Install from sources](https://opennmt.net/CTranslate2/installation.html#install-from-sources).

### Running the tests

#### C++

To enable the C++ tests, you should configure the project with `cmake -DBUILD_TESTS=ON`. The binary `tests/ctranslate2_test` runs all tests using [Google Test](https://github.com/google/googletest). It expects the path to the test data as argument:

```bash
./tests/ctranslate2_test ../tests/data
```

#### Python

The Python tests can be run with `pytest`:

```bash
cd python
pip install -r tests/requirements.txt
pytest tests/
```

The code should also be checked with `black` (automatic formatting), `isort` (imports ordering), and `flake8` (code checking):

```bash
black .
isort .
flake8 .
```

### Measuring performance

You should make sure that new changes do not negatively impact the general performance. The translation client has some options to measure the performance.

#### Translation throughput

The command line option `--log_throughput` reports the *tokens generated per second* on the standard error output. This is the recommended metric to compare different runs (higher is better).

#### Execution profile

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

### Implementation details

#### `StorageView` class

CTranslate2 uses [row-major](https://en.wikipedia.org/wiki/Row-_and_column-major_order) storages, usually encapsulated in the `StorageView` class. This class acts like a tensor representation but without the mathematical semantics. It is convenience wrapper to view a buffer of data in a particular shape, and provides methods to resize, reshape, and copy data. The underlying storage has a type (e.g. `float`) and a location (e.g. GPU #1) which are both resolved at runtime.

To maximize performance, the implementation avoid new allocations when possible:

* no reallocation occurs when resizing the storage to a smaller size
* caching allocators are used to reuse previously allocated buffers

#### Abstraction levels

From lowest to highest level:

* *kernels*: low-level compute functions (e.g. CUDA implementation of Softmax)
* *primitives*: basic vector and matrix processing functions (e.g. addition of two C arrays)
* *ops*: neural network operations (e.g. Softmax, Gemm, etc.)
* *layers*: stateful neural network layers (e.g. `Dense`, `LayerNorm`, etc.)
* *models*: collection of neural network layers and weights (e.g. `Transformer`)
* *replicas*: runnable instances of a model
* *replicas pool*: thread pool of model replicas

#### Ops

Ops define the basic neural network operations. The op interface is sometimes inspired by the [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md).

Their implementation typically require multiple source files:

```text
include/ctranslate2/ops/my_op.h  # Op interface
src/ops/my_op.cc                 # Input checks and dispatch based on device and type.
src/ops/my_op_cpu.cc             # CPU-specific implementation
src/ops/my_op_gpu.cu             # CUDA-specific implementation
```

In particular, no compilation flags should be used in the header file to make it easy to use the project as a library.

## Maintenance

### Updating the Python build matrix

Binary Python wheels are built for multiple Python versions in the `build-python-wheels` GitHub Actions job. The list of Python versions is defined by the intersection of:

* `python_requires` in file `python/setup.py`
* the default build list in [`cibuildwheel`](https://github.com/pypa/cibuildwheel)

Building wheels for a new Python version usually means updating the `cibuildwheel` version in `.github/workflows/ci.yml`. See for example commit [8f4c7ade1](https://github.com/OpenNMT/CTranslate2/commit/8f4c7ade14cba114c8acad2cc700edc1704c8396).

### CUDA support in Python wheels

Python wheels for Linux and Windows are compiled against NVIDIA libraries to support GPU execution.

To limit the size of the packages pushed to PyPI, some libraries are not included in the package and are dynamically loaded at runtime with `dlopen` (or `LoadLibraryA` on Windows).

* `libcudart_static.a` (statically linked)
* `libcublas.so.12` (dlopened at runtime in [`cublas_stub.cc`](https://github.com/OpenNMT/CTranslate2/blob/master/src/cuda/cublas_stub.cc))
* `libcudnn.so.8` (dynamically linked)
  * `libcudnn_ops_infer.so.8` (dlopened at runtime by `libcudnn.so.8`)
  * `libcudnn_cnn_infer.so.8` (dlopened at runtime by `libcudnn.so.8`)

One of the benefits of this dynamic loading is that multiple versions of cuBLAS and cuDNN are supported by the same binary. In particular, users can install any CUDA 12.x version as long as it provides `libcublas.so.12`.

The Python library only support CUDA 12.x. C++ source code is always compatible with CUDA 11, possible to use CUDA 11 libraries during compilation to create CUDA 11.x support wheel.

### Updating other dependencies

Updating dependencies such as oneMKL or oneDNN can fix security issues, improve performance, or enable new features. Most dependencies were already updated at least once. Search the commit history to see how it was done before.

If a dependency needs an update, it is particularly important that it is updated consistently for all binaries and platforms where it is used.

## Release procedure

1. Add the release note for the new version in `CHANGELOG.md`
1. Update the version number in `python/ctranslate2/version.py`
1. Tag the latest commit in the format `vX.Y.Z`
1. When the release pipeline is finished, create a new release on GitHub and copy the note from `CHANGELOG.md`

### Managing PyPI project size limit

Projects on PyPI have a size limit. The default limit is 10GB and [we already requested](https://github.com/pypi/support/issues/1480) an increase to 20GB in the past. Because increase requests can take several months to be accepted, we now try to work with this 20GB limit.

So older releases need to be regularly deleted on PyPI to make room for new releases. **However, make sure to keep the latest release of each major version.**

Here's the process to delete a release:

1. Download the releases to be deleted (see an example script below)
1. Upload the wheels in the `ctranslate2-wheels` bucket of the OpenNMT AWS S3 account
1. Delete the release on PyPI

**Example script to download the wheels of a release:**

```bash
#! /bin/bash

VERSION="1.20.0"

wget https://pypi.org/simple/ctranslate2/ -O /tmp/ctranslate2.html
mkdir -p /tmp/wheels
grep -F "$VERSION" /tmp/ctranslate2.html | sed -E 's/.*<a href="([^"]+)".*/\1/i' | xargs wget -P /tmp/wheels
```
