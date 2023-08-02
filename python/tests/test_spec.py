import numpy as np
import pytest
import test_utils

import ctranslate2

from ctranslate2.converters import utils as conversion_utils
from ctranslate2.specs import common_spec, transformer_spec
from ctranslate2.specs.model_spec import OPTIONAL, index_spec


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
    assert spec.a.dtype == "float32"
    assert spec.b.dtype == "float16"
    assert spec.c.dtype == "int32"
    assert spec.d == OPTIONAL
    assert spec.e.a.dtype == "float16"
    assert test_utils.array_equal(spec.f.numpy(), np.int8(1))
    assert test_utils.array_equal(
        spec.g.numpy(), np.array([104, 101, 108, 108, 111], dtype=np.int8)
    )

    with pytest.raises(AttributeError, match="Attribute z does not exist"):
        spec.z = True


def test_layer_spec_validate_unset():
    class SubSpec(ctranslate2.specs.LayerSpec):
        def __init__(self):
            self.attr_1 = None

    class Spec(ctranslate2.specs.LayerSpec):
        def __init__(self):
            self.attr_1 = np.zeros([5], dtype=np.float32)
            self.attr_2 = None
            self.attr_3 = SubSpec()

    spec = Spec()

    with pytest.raises(ValueError, match="attr_2\nattr_3.attr_1"):
        spec.validate()


def test_layer_spec_optimize():
    class SubSpec(ctranslate2.specs.LayerSpec):
        def __init__(self):
            self.a = np.ones([6], dtype=np.float32)
            self.weight = np.ones([5, 4], dtype=np.float32)
            self.weight_scale = OPTIONAL

    class Spec(ctranslate2.specs.LayerSpec):
        def __init__(self):
            self.a = np.ones([5], dtype=np.float32)
            self.b = np.ones([5], dtype=np.float32)
            self.c = np.zeros([5], dtype=np.int32)
            self.d = np.dtype("float32").type(3.14)
            self.sub = SubSpec()

    spec = Spec()
    spec.validate()
    spec.optimize(quantization="int16")
    assert spec.a.dtype == "float32"
    assert spec.b == "a"
    assert spec.c.dtype == "int32"
    assert spec.d.dtype == "float32"
    assert spec.sub.weight.dtype == "int16"
    assert spec.sub.weight_scale.dtype == "float32"

    spec = Spec()
    spec.validate()
    spec.optimize(quantization="float16")
    assert spec.a.dtype == "float16"
    assert spec.b == "a"
    assert spec.c.dtype == "int32"
    assert spec.d.dtype == "float32"
    assert spec.sub.weight.dtype == "float16"
    assert spec.sub.a.dtype == "float16"

    spec = Spec()
    spec.validate()
    with pytest.raises(ValueError, match="not a valid quantization type"):
        spec.optimize(quantization="int32")


def test_int8_quantization():
    class Spec(ctranslate2.specs.LayerSpec):
        def __init__(self):
            self.weight = np.array([[-10, -3, 5, 2], [0, 0, 0, 0]], dtype=np.float32)
            self.weight_scale = OPTIONAL

    spec = Spec()
    spec.validate()
    spec.optimize(quantization="int8")
    assert test_utils.array_equal(
        spec.weight.numpy(),
        np.array([[-127, -38, 64, 25], [0, 0, 0, 0]], dtype=np.int8),
    )
    assert test_utils.array_equal(
        spec.weight_scale.numpy(), np.array([12.7, 1], dtype=np.float32)
    )


@pytest.mark.parametrize(
    "quantization,expected_weight,expected_weight_scale,expected_bias",
    [
        (
            None,
            np.array([[-10, -3, 5, 2]], dtype=np.float16),
            None,
            np.array([4], dtype=np.float16),
        ),
        (
            "float32",
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
            np.array([4], dtype=np.float16),
        ),
        (
            "int8_float16",
            np.array([[-127, -38, 64, 25]], dtype=np.int8),
            np.array([12.7], dtype=np.float32),
            np.array([4], dtype=np.float16),
        ),
        (
            "int8_float32",
            np.array([[-127, -38, 64, 25]], dtype=np.int8),
            np.array([12.7], dtype=np.float32),
            np.array([4], dtype=np.float32),
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

    assert test_utils.array_equal(spec.weight.numpy(), expected_weight)
    assert test_utils.array_equal(spec.bias.numpy(), expected_bias)

    # Check the weights were not copied or converted.
    if quantization in (None, "float16"):
        assert spec.weight.numpy() is weight
        assert spec.bias.numpy() is bias
    elif quantization in ("int8", "int8_float16"):
        assert spec.bias.numpy() is bias

    if expected_weight_scale is None:
        assert spec.weight_scale == OPTIONAL
    else:
        assert test_utils.array_equal(spec.weight_scale.numpy(), expected_weight_scale)


def test_index_spec():
    spec = ctranslate2.specs.TransformerSpec.from_config(6, 8)
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
    assert test_utils.array_equal(spec.bias[:64], np.zeros([64], dtype=np.float32))
    assert test_utils.array_equal(spec.bias[64:128], np.ones([64], dtype=np.float32))
    assert test_utils.array_equal(spec.bias[128:], np.zeros([64], dtype=np.float32))


@test_utils.skip_on_windows
def test_fuse_linear_torch():
    import torch

    layers = []
    for _ in range(3):
        spec = common_spec.LinearSpec()
        spec.weight = torch.zeros([64, 64], dtype=torch.float32)
        spec.bias = torch.zeros([64], dtype=torch.float32)
        layers.append(spec)

    spec = common_spec.LinearSpec()
    conversion_utils.fuse_linear(spec, layers)
    assert spec.weight.shape[0] == 64 * 3
    assert spec.bias.shape[0] == 64 * 3


@test_utils.skip_on_windows
def test_smooth_activation_torch():
    import torch

    layer_norm = common_spec.LayerNormSpec()
    layer_norm.beta = torch.rand([64], dtype=torch.float16)
    layer_norm.gamma = torch.rand([64], dtype=torch.float16)

    linear = common_spec.LinearSpec()
    linear.weight = torch.rand([64, 64], dtype=torch.float16)

    activation_scales = torch.rand([64], dtype=torch.float32)

    # Just check that no error is raised.
    conversion_utils.smooth_activation(layer_norm, linear, activation_scales)


@test_utils.skip_on_windows
@pytest.mark.parametrize("variable_dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize(
    "quantization,expected_weight_dtype,expected_bias_dtype",
    [
        (None, None, None),
        ("int8", "int8", None),
        ("int8_float32", "int8", "float32"),
        ("int8_float16", "int8", "float16"),
        ("int8_bfloat16", "int8", "bfloat16"),
        ("int16", "int16", "float32"),
        ("float16", "float16", "float16"),
        ("bfloat16", "bfloat16", "bfloat16"),
        ("float32", "float32", "float32"),
    ],
)
def test_torch_variables(
    tmp_dir, variable_dtype, quantization, expected_weight_dtype, expected_bias_dtype
):
    import torch

    if expected_weight_dtype is None:
        expected_weight_dtype = variable_dtype
    if expected_bias_dtype is None:
        expected_bias_dtype = variable_dtype

    variable_dtype = getattr(torch, variable_dtype)

    class TorchModel(ctranslate2.specs.ModelSpec):
        def __init__(self):
            super().__init__()
            self.dense = common_spec.LinearSpec()
            self.dense.weight = torch.ones([16, 4], dtype=variable_dtype)
            self.dense.bias = torch.ones([16], dtype=variable_dtype)

        @property
        def name(self):
            return "TorchModel"

    model = TorchModel()
    model.validate()
    model.optimize(quantization)

    variables = model.variables()
    assert variables["dense/weight"].dtype == expected_weight_dtype
    assert variables["dense/bias"].dtype == expected_bias_dtype

    model.save(tmp_dir)
