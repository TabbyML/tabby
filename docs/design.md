# Design

This document describes some design elements of the CTranslate2 implementation, making it easier for developers to get started.

## General

### Model specification

A model specification defines the variables expected by each model.

In the Python code, it is represented as nested `LayerSpec` objects where intermediate objects define variable scopes and leaf objects define variables name and value. This is similar to how you would define a model in PyTorch or TensorFlow.

The final structure defines the full name of each variable that the C++ code should read when building the model. For example, a variable that can be accessed with `root.encoder.embeddings.weight` will have for name `encoder/embeddings/weight`.

Changes in this structure are tracked by a revision number (see next section).

### Model serialization

The model serialization is defined in the Python file `python/ctranslate2/specs/model_spec.py`. It is a simple binary serialization that is easy and fast to load from C++.

Converted models have 2 levels of versioning to manage backward compatibility:

1. Binary version: the structure of the binary file
2. Model specification revision: the variable names expected by each model.

For example, adding a new field in the binary file will increment (1), but changing a variable name will increment (2).

## C++ engine

### Storage

CTranslate2 uses [row-major](https://en.wikipedia.org/wiki/Row-_and_column-major_order) storages, usually encapsulated in the `StorageView` class. This class acts like a tensor representation but without the mathematical semantics. It is convenience wrapper to view a buffer of data in a particular shape, and provides methods to resize, reshape, and copy data.

### Abstraction levels

* *primitives*: low-level compute functions, specialized depending on the data type and target device.
* *ops*: neural network operations (e.g. Softmax, Gemm, etc.) following (possibly partially) the [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md).
* *layers*: stateful neural network layers like `Dense`, `LayerNorm`, etc.
* *models*: collection of neural network layers to achieve a certain tasks (e.g. `Transformer` for NMT)
* *translators*: high-level class using a model to implement the text translation logic
* *translators pool*: pool of parallel translators sharing the same model

### Ops

Ops typically require multiple source files:

```text
include/ctranslate2/ops/my_op.h  # Op interface
src/ops/my_op.cc                 # Input checks and dispatch based on device and type.
src/ops/my_op_cpu.cc             # CPU-specific implementation
src/ops/my_op_gpu.cu             # CUDA-specific implementation
```

In particular, no compilation flags is used in the header file.
