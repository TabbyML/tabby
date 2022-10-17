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
