import enum

from ctranslate2.specs import model_spec


# This enum should match the C++ equivalent in include/ctranslate2/ops/activation.h.
class Activation(enum.IntEnum):
    RELU = 0
    GELU = 1


class LayerNormSpec(model_spec.LayerSpec):
    def __init__(self):
        self.gamma = None
        self.beta = None


class LinearSpec(model_spec.LayerSpec):
    def __init__(self):
        self.weight = None
        self.bias = model_spec.OPTIONAL


class EmbeddingsSpec(model_spec.LayerSpec):
    def __init__(self):
        self.weight = None
        self.multiply_by_sqrt_depth = False
