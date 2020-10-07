# Design

This document describes some design elements of the CTranslate2 implementation, making it easier for developers to get started.

## Checkpoint conversion

The core CTranslate2 implementation is framework-agnostic and does not depend on PyTorch or TensorFlow. The logic that is specific to each training framework is moved to a checkpoint conversion step to produce a unified model representation.

### Model specification

A model specification defines the structures and names of the model weights. Converters should fill out this specification with weights comming from a trained model.

In the Python code, a model specification is represented as nested `LayerSpec` objects, where intermediate objects define weights scopes and leaf objects define the weights name and value. This is similar to how you would define a model in PyTorch (using `nn.Module`) or TensorFlow (using `tf.Module`).

The final structure defines the full name of each weight that the C++ code should read when building the model. For example, a weight that can be accessed with `root.encoder.embeddings.weight` (where `root` is the top-level `LayerSpec` object) will have for name `encoder/embeddings/weight` in the serialized model.

Changes in this structure are tracked by a revision number (see next section).

### Model serialization

The model serialization is defined in the Python file `python/ctranslate2/specs/model_spec.py`. It is a simple binary serialization that is easy and fast to load from C++.

Converted models have 2 levels of versioning to manage backward compatibility:

1. Binary version: the structure of the binary file
2. Model specification revision: the variable names expected by each model.

For example, adding a new field in the binary file will increment (1), but changing a variable name will increment (2).

## C++ engine

### Storage

CTranslate2 uses [row-major](https://en.wikipedia.org/wiki/Row-_and_column-major_order) storages, usually encapsulated in the `StorageView` class. This class acts like a tensor representation but without the mathematical semantics. It is convenience wrapper to view a buffer of data in a particular shape, and provides methods to resize, reshape, and copy data. The underlying storage has a type (e.g. `float`) and a location (e.g. GPU #1) which are both resolved at runtime.

To maximize performance, the implementation avoid new allocations when possible:

* no reallocation occurs when resizing the storage to a smaller size
* caching allocators are used to reuse previously allocated buffers

### Abstraction levels

* *primitives*: low-level compute functions, specialized depending on the data type and target device.
* *ops*: neural network operations (e.g. Softmax, Gemm, etc.)
* *layers*: stateful neural network layers like `Dense`, `LayerNorm`, etc.
* *models*: collection of neural network layers to achieve a certain tasks (e.g. `Transformer` for NMT)
* *translators*: high-level class using a model to implement the text translation logic
* *translators pool*: pool of parallel translators sharing the same model

### Multi architectures and backends

CTranslate2 can integrate multiple backends and architectures in a single binary. The goal is "build once, run everywhere."

The backend can be selected at runtime by the user (e.g. CPU or GPU) or automatically based on the system information. For example, CTranslate2 will not use Intel MKL on a non Intel CPU if a fallback is available. Similarly the CPU architecture is detected at runtime (e.g. AVX) and the fastest code branch is executed accordingly.

### Ops

Ops define the basic neural network operations. Whenever possible, they follow the [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md).

Their implementation typically require multiple source files:

```text
include/ctranslate2/ops/my_op.h  # Op interface
src/ops/my_op.cc                 # Input checks and dispatch based on device and type.
src/ops/my_op_cpu.cc             # CPU-specific implementation
src/ops/my_op_gpu.cu             # CUDA-specific implementation
```

In particular, no compilation flags should be used in the header file to make it easy to use the project as a library.

## Python wrapper

The Python wrapper uses [pybind11](https://github.com/pybind/pybind11/). On each new release, the CI builds and pushes `manylinux2010` wheels to PyPI.
