import numpy as np


def fuse_linear(spec, layers):
    if not layers:
        raise ValueError("Cannot fuse linear layers: at least one layer is required")

    spec.weight = np.concatenate([layer.weight for layer in layers])

    bias_dtype = None
    for layer in layers:
        if layer.has_bias():
            bias_dtype = layer.bias.dtype
            break

    if bias_dtype is not None:
        spec.bias = np.concatenate(
            [
                layer.bias
                if layer.has_bias()
                else np.zeros([layer.weight.shape[0]], dtype=bias_dtype)
                for layer in layers
            ]
        )


def smooth_activation(layer_norm, linear, activation_scales):
    """Applies the activation smoothing technique described in
    https://github.com/mit-han-lab/smoothquant.
    """

    weight_scales = np.amax(np.absolute(linear.weight), axis=0)
    weight_scales = np.maximum(weight_scales, 1e-5)

    activation_scales = activation_scales.astype(weight_scales.dtype)

    scales = np.sqrt(activation_scales / weight_scales)
    scales = np.maximum(scales, 1e-5)

    layer_norm.gamma /= scales
    layer_norm.beta /= scales

    linear.weight *= np.expand_dims(scales, 0)


def raise_unsupported(reasons):
    message = (
        "The model you are trying to convert is not supported by CTranslate2. "
        "We identified the following reasons:\n"
    )
    for reason in reasons:
        message += "\n- " + reason
    raise ValueError(message)


class ConfigurationChecker:
    def __init__(self):
        self._unsupported_reasons = []

    def __call__(self, assert_condition, error_message):
        if not assert_condition:
            self._unsupported_reasons.append(error_message)

    def validate(self):
        if self._unsupported_reasons:
            raise_unsupported(self._unsupported_reasons)
