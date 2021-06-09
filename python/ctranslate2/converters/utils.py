import numpy as np


def fuse_linear(spec, layers):
    spec.weight = np.concatenate([layer.weight for layer in layers])
    spec.bias = np.concatenate([layer.bias for layer in layers])


def raise_unsupported(reasons):
    message = (
        "The model you are trying to convert is not supported by CTranslate2. "
        "We identified the following reasons:\n"
    )
    for reason in reasons:
        message += "\n- " + reason
    raise ValueError(message)
