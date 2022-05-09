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
pytest tests/test.py
```

The code should also be checked with `flake8` and reformatted with `black`:

```bash
black .
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

* *primitives*: low-level compute functions, specialized depending on the data type and target device.
* *ops*: neural network operations (e.g. Softmax, Gemm, etc.)
* *layers*: stateful neural network layers like `Dense`, `LayerNorm`, etc.
* *models*: collection of neural network layers to achieve a certain tasks (e.g. `Transformer` for NMT)
* *translators*: high-level class using a model to implement the text translation logic
* *translators pool*: pool of parallel translators sharing the same model

#### Ops

Ops define the basic neural network operations. Whenever possible, they follow the [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md).

Their implementation typically require multiple source files:

```text
include/ctranslate2/ops/my_op.h  # Op interface
src/ops/my_op.cc                 # Input checks and dispatch based on device and type.
src/ops/my_op_cpu.cc             # CPU-specific implementation
src/ops/my_op_gpu.cu             # CUDA-specific implementation
```

In particular, no compilation flags should be used in the header file to make it easy to use the project as a library.
