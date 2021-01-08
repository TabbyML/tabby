import numpy as np


def fuse_linear(spec, layers):
    spec.weight = np.concatenate([layer.weight for layer in layers])
    spec.bias = np.concatenate([layer.bias for layer in layers])
