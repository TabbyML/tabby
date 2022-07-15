import enum

import numpy as np

from ctranslate2.specs import model_spec


# This enum should match the C++ equivalent in include/ctranslate2/ops/activation.h.
class Activation(enum.IntEnum):
    """Activation type."""

    RELU = 0
    GELU = 1
    SWISH = 2


# This enum should match the C++ equivalent in include/ctranslate2/layers/common.h.
class EmbeddingsMerge(enum.IntEnum):
    """Merge strategy for factors embeddings."""

    CONCAT = 0
    ADD = 1


class LayerNormSpec(model_spec.LayerSpec):
    def __init__(self):
        self.gamma = None
        self.beta = None


class LinearSpec(model_spec.LayerSpec):
    def __init__(self):
        self.weight = None
        self.weight_scale = model_spec.OPTIONAL
        self.bias = model_spec.OPTIONAL

    def has_bias(self):
        return isinstance(self.bias, np.ndarray)


class EmbeddingsSpec(model_spec.LayerSpec):
    def __init__(self):
        self.weight = None
        self.weight_scale = model_spec.OPTIONAL
