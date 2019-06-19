# Design

This document describes some design elements of the CTranslate2 implementation, making it easier for developers to get started.

## Storage

CTranslate2 uses [row-major](https://en.wikipedia.org/wiki/Row-_and_column-major_order) storages, usually encapsulated in the `StorageView` class. This class acts like a tensor representation but without the mathematical semantics. It is convenience wrapper to view a buffer of data in a particular shape, and provides methods to resize, reshape, and copy data.

## Abstraction levels

* *primitives*: low-level compute functions, specialized depending on the data type and target device.
* *ops*: neural network operations (e.g. Softmax, Gemm, etc.) following (possibly partially) the [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md).
* *layers*: stateful neural network layers like `Dense`, `LayerNorm`, etc.
* *models*: collection of neural network layers to achieve a certain tasks (e.g. `Transformer` for NMT)
* *translators*: high-level class using a model to implement the text translation logic
* *translators pool*: pool of parallel translators sharing the same model
