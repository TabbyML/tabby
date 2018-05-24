# WIP CTranslate2

This is an early preview of some CTranslate2 components:

## `storage_view.h`

The `StorageView` class is a light wrapper around an allocated buffer to give it a sense of shape.

* It can be resized, reshaped, copied, or assigned
* It can view an existing buffer to avoid memory copy
* The buffer can be of any type and casting is supported
* Allocation is aligned by default to 64 bytes: wasted space is minimal and it is required when working with intrinsics up to AVX512

## `ops.h`

This file defines operations as close as possible to the ONNX specs.

* The inputs/outputs are templated `StorageView` instances.

## `compute.h`

This file defines low-level compute primitive on raw arrays.

* All primitives are templated and most of them come with a generic implementation (e.g. a simple loop over the array)
* Optimizations are implemented via template specialization (e.g. for `float` arrays, MKL primitives are used)


---

* `convert.py`
* `main.cc`
* `model.h`
* `vocabulary.h`

are Transformer-specific implementations making use of the files above.
