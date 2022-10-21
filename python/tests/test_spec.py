import numpy as np
import pytest

import ctranslate2

from ctranslate2.converters import utils as conversion_utils
from ctranslate2.specs import common_spec, transformer_spec
from ctranslate2.specs.model_spec import OPTIONAL, index_spec


def _array_equal(a, b):
    return a.dtype == b.dtype and np.array_equal(a, b)


def test_layer_spec_validate():
    class SubSpec(ctranslate2.specs.LayerSpec):
        def __init__(self):
            self.a = np.ones([5], dtype=np.float16)

    class Spec(ctranslate2.specs.LayerSpec):
        def __init__(self):
            self.a = np.zeros([5], dtype=np.float32)
            self.b = np.zeros([5], dtype=np.float16)
            self.c = np.zeros([5], dtype=np.int32)
            self.d = OPTIONAL
            self.e = SubSpec()
            self.f = True
            self.g = "hello"

    spec = Spec()
    spec.validate()
    assert spec.a.dtype == np.float32
    assert spec.b.dtype == np.float16
    assert spec.c.dtype == np.int32
    assert spec.d == OPTIONAL
    assert spec.e.a.dtype == np.float16
    assert _array_equal(spec.f, np.int8(1))
    assert _array_equal(spec.g, np.array([104, 101, 108, 108, 111], dtype=np.int8))

    with pytest.raises(AttributeError, match="Attribute z does not exist"):
        spec.z = True


def test_layer_spec_optimize():
    class SubSpec(ctranslate2.specs.LayerSpec):
        def __init__(self):
            self.a = np.ones([6], dtype=np.float32)

    class Spec(ctranslate2.specs.LayerSpec):
        def __init__(self):
            self.a = np.ones([5], dtype=np.float32)
            self.b = np.ones([5], dtype=np.float32)
            self.c = np.zeros([5], dtype=np.int32)
            self.d = np.dtype("float32").type(3.14)
            self.weight = np.ones([5, 4], dtype=np.float32)
            self.weight_scale = OPTIONAL
            self.sub = SubSpec()

    spec = Spec()
    spec.optimize(quantization="int16")
    assert spec.a.dtype == np.float32
    assert spec.b == "a"
    assert spec.c.dtype == np.int32
    assert spec.d.dtype == np.float32
    assert spec.weight.dtype == np.int16
    assert spec.weight_scale.dtype == np.float32

    spec = Spec()
    spec.optimize(quantization="float16")
    assert spec.a.dtype == np.float16
    assert spec.b == "a"
    assert spec.c.dtype == np.int32
    assert spec.d.dtype == np.float32
    assert spec.weight.dtype == np.float16
    assert spec.sub.a.dtype == np.float16


def test_int8_quantization():
    class Spec(ctranslate2.specs.LayerSpec):
        def __init__(self):
            self.weight = np.array([[-10, -3, 5, 2], [0, 0, 0, 0]], dtype=np.float32)
            self.weight_scale = OPTIONAL

    spec = Spec()
    spec.optimize(quantization="int8")
    assert _array_equal(
        spec.weight, np.array([[-127, -38, 64, 25], [0, 0, 0, 0]], dtype=np.int8)
    )
    assert _array_equal(spec.weight_scale, np.array([12.7, 1], dtype=np.float32))


@pytest.mark.parametrize(
    "quantization,expected_weight,expected_weight_scale,expected_bias",
    [
        (
            None,
            np.array([[-10, -3, 5, 2]], dtype=np.float32),
            None,
            np.array([4], dtype=np.float32),
        ),
        (
            "float16",
            np.array([[-10, -3, 5, 2]], dtype=np.float16),
            None,
            np.array([4], dtype=np.float16),
        ),
        (
            "int8",
            np.array([[-127, -38, 64, 25]], dtype=np.int8),
            np.array([12.7], dtype=np.float32),
            np.array([4], dtype=np.float32),
        ),
        (
            "int8_float16",
            np.array([[-127, -38, 64, 25]], dtype=np.int8),
            np.array([12.7], dtype=np.float32),
            np.array([4], dtype=np.float16),
        ),
        (
            "int16",
            np.array([[-1024, -307, 512, 205]], dtype=np.int16),
            np.float32(102.4),
            np.array([4], dtype=np.float32),
        ),
    ],
)
def test_fp16_weights(
    quantization, expected_weight, expected_weight_scale, expected_bias
):
    class Spec(ctranslate2.specs.LayerSpec):
        def __init__(self, weight, bias):
            self.weight = weight
            self.weight_scale = OPTIONAL
            self.bias = bias

    weight = np.array([[-10, -3, 5, 2]], dtype=np.float16)
    bias = np.array([4], dtype=np.float16)

    spec = Spec(weight, bias)
    spec.validate()
    spec.optimize(quantization=quantization)

    assert _array_equal(spec.weight, expected_weight)
    assert _array_equal(spec.bias, expected_bias)

    # Check the weights were not copied or converted.
    if quantization == "float16":
        assert spec.weight is weight
        assert spec.bias is bias
    elif quantization == "int8_float16":
        assert spec.bias is bias

    if expected_weight_scale is None:
        assert spec.weight_scale == OPTIONAL
    else:
        assert _array_equal(spec.weight_scale, expected_weight_scale)


def test_index_spec():
    spec = ctranslate2.specs.TransformerSpec(6, 8)
    assert isinstance(
        index_spec(spec, "encoder/layer_5"),
        transformer_spec.TransformerEncoderLayerSpec,
    )
    assert isinstance(
        index_spec(spec, "encoder/layer_5/ffn"), transformer_spec.FeedForwardSpec
    )


def test_fuse_linear_no_bias():
    layers = []
    for _ in range(3):
        spec = common_spec.LinearSpec()
        spec.weight = np.zeros([64, 64], dtype=np.float32)
        layers.append(spec)

    spec = common_spec.LinearSpec()
    conversion_utils.fuse_linear(spec, layers)
    assert spec.weight.shape[0] == 64 * 3
    assert spec.bias == OPTIONAL

    spec = common_spec.LinearSpec()
    layers[1].bias = np.ones([64], dtype=np.float32)
    conversion_utils.fuse_linear(spec, layers)
    assert _array_equal(spec.bias[:64], np.zeros([64], dtype=np.float32))
    assert _array_equal(spec.bias[64:128], np.ones([64], dtype=np.float32))
    assert _array_equal(spec.bias[128:], np.zeros([64], dtype=np.float32))
