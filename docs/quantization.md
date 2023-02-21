# Quantization

Quantization is a technique that can reduce the model size and accelerate its execution with little to no degradation in accuracy. CTranslate2 supports the most common types:

* 8-bit integers (INT8)
* 16-bit integers (INT16)
* 16-bit floating points (FP16)

```{tip}
See the benchmark results in the main [README](https://github.com/OpenNMT/CTranslate2#benchmarks) to compare the performance and memory usage with and without quantization.
```

## Quantize on model conversion

Enabling the quantization when converting the model is helpful to reduce its size on disk. The converters expose the option `quantization` that accepts the following values:

* `int8`
* `int8_float16`
* `int16`
* `float16`

For example,

```bash
ct2-opennmt-py-converter --model_path model.pt --quantization int8 --output_dir ct2_model
```

```{note}
Whatever quantization type is selected here, the runtime ensures the model can be loaded and executed efficiently. This implies the model weights are possibly converted to another type when the model is loaded, see {ref}`quantization:implicit type conversion on load`.
```

For reference, the table below compares the model size on disk for a base Transformer model without shared embeddings and a vocabulary of size 32k:

| Quantization | Model size |
| --- | --- |
| None | 364MB |
| `int16` | 187MB |
| `float16` | 182MB |
| `int8` | 100MB |
| `int8_float16` | 95MB |

## Quantize on model loading

Quantization can also be enabled or changed when loading the model. The translator exposes the option `compute_type` that accepts the following values:

* `default`: keep the same quantization that was used during model conversion (see {ref}`quantization:implicit type conversion on load` for exceptions)
* `auto`: use the fastest computation type that is supported on this system and device
* `int8`
* `int8_float16`
* `int16`
* `float16`
* `float32`

For example,

```python
translator = ctranslate2.Translator(model_path, compute_type="int8")
```

```{tip}
Conversions between all types are supported. For example, you can convert a model with `quantization="int8"` and then execute in full precision with `compute_type="float32"`.
```

## Implicit type conversion on load

By default, the runtime tries to use the type that is saved in the converted model as the computation type. However, if the current platform or backend do not support optimized execution for this computation type (e.g. `int16` is not optimized on GPU), then the library converts the model weights to another optimized type. The tables below document the fallback types in prebuilt binaries:

**On CPU:**

| Architecture | int8 | int8_float16 | int16 | float16 |
| --- | --- | --- | --- | --- |
| x86-64 (Intel) | int8 | int8 | int16 | float32 |
| x86-64 (other) | int8 | int8 | int8 | float32 |
| AArch64/ARM64 (Apple) | float32 | float32 | float32 | float32 |
| AArch64/ARM64 (other) | int8 | int8 | int8 | float32 |

**On GPU:**

| Compute Capability | int8 | int8_float16 | int16 | float16 |
| --- | --- | --- | --- | --- |
| >= 7.0 | int8 | int8_float16 | float16 | float16 |
| 6.2 | float32 | float32 | float32 | float32 |
| 6.1 | int8 | int8 | float32 | float32 |
| <= 6.0 | float32 | float32 | float32 | float32 |

```{tip}
You can get more information about the detected capabilities of your system by enabling the info logs (set the environment variable `CT2_VERBOSE=1` or call ``ctranslate2.set_log_level(logging.INFO)``).

The supported compute types can also be queried at runtime with the Python function [`ctranslate2.get_supported_compute_types`](python/ctranslate2.get_supported_compute_types.rst).
```

## Supported types

### 8-bit integers (`int8`)

**Supported on:**

* NVIDIA GPU with Compute Capability >= 7.0 or Compute Capability 6.1
* x86-64 CPU with the Intel MKL or oneDNN backends
* AArch64/ARM64 CPU with the Ruy backend

The implementation applies the equation from [Wu et al. 2016](https://arxiv.org/abs/1609.08144) to quantize the weights of the embedding and linear layers:

```text
scale[i] = 127 / max(abs(W[i,:]))

WQ[i,j] = round(scale[i] * W[i,j])
```

```{note}
This formula corresponds to a symmetric quantization (absolute maximum of the input range instead of separate min/max values).
```

### 16-bit integers (`int16`)

**Supported on:**

* Intel CPU with the Intel MKL backend

The implementation follows the work by [Devlin 2017](https://arxiv.org/abs/1705.01991). By default we use one quantization scale per layer. The scale is defined as:

```text
scale = 2^10 / max(abs(W))
```

As suggested by the author, the idea is to use 10 bits for the input so that the multiplication is 20 bits which gives 12 bits left for accumulation.

Similar to the `int8` quantization, only the weights of the embedding and linear layers are quantized to 16-bit integers.

### 16-bit floating points (`float16`)

**Supported on:**

* NVIDIA GPU with Compute Capability >= 7.0

In this mode, all model weights are stored in half precision and all layers are run in half precision.

### Mixed 8-bit integers and 16-bit floating points (`int8_float16`)

**Supported on:**

* NVIDIA GPU with Compute Capability >= 7.0

This mode is the same as `int8`, but all non quantized layers are run in FP16 instead of FP32.
