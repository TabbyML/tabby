"""Declares specification of the Transformer model."""

import numpy as np

from ctranslate2.specs import model_spec


class TransformerSpec(model_spec.ModelSpec):
    """Describes a Transformer model.

    The specification is invariant to hidden dimensions but requires to
    explicitly set the number of layers and attention heads.
    """
    def __init__(self, num_layers, num_heads):
        self.num_heads = np.dtype("int8").type(num_heads)
        self.encoder = TransformerEncoderSpec(num_layers)
        self.decoder = TransformerDecoderSpec(num_layers)

    @property
    def revision(self):
        return 3

    @property
    def source_vocabulary_size(self):
        return self.encoder.embeddings.weight.shape[0]

    @property
    def target_vocabulary_size(self):
        return self.decoder.embeddings.weight.shape[0]

class TransformerEncoderSpec(model_spec.LayerSpec):
    def __init__(self, num_layers):
        self.embeddings = EmbeddingsSpec()
        self.position_encodings = PositionEncoderSpec()
        self.layer_norm = LayerNormSpec()
        self.layer = [TransformerEncoderLayerSpec() for _ in range(num_layers)]

class TransformerDecoderSpec(model_spec.LayerSpec):
    def __init__(self, num_layers):
        self.embeddings = EmbeddingsSpec()
        self.position_encodings = PositionEncoderSpec()
        self.layer_norm = LayerNormSpec()
        self.projection = LinearSpec()
        self.layer = [
            TransformerDecoderLayerSpec() for _ in range(num_layers)]

class TransformerEncoderLayerSpec(model_spec.LayerSpec):
    def __init__(self):
        self.self_attention = MultiHeadAttentionSpec(self_attention=True)
        self.ffn = FeedForwardSpec()

class TransformerDecoderLayerSpec(model_spec.LayerSpec):
    def __init__(self):
        self.self_attention = MultiHeadAttentionSpec(self_attention=True)
        self.attention = MultiHeadAttentionSpec()
        self.ffn = FeedForwardSpec()

class MultiHeadAttentionSpec(model_spec.LayerSpec):
    def __init__(self, self_attention=False):
        self.layer_norm = LayerNormSpec()
        if self_attention:
            num_projections = 2
        else:
            num_projections = 3
        self.linear = [LinearSpec() for _ in range(num_projections)]

class FeedForwardSpec(model_spec.LayerSpec):
    def __init__(self):
        self.layer_norm = LayerNormSpec()
        self.linear_0 = LinearSpec()
        self.linear_1 = LinearSpec()

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

class PositionEncoderSpec(model_spec.LayerSpec):
    def __init__(self):
        self.encodings = model_spec.OPTIONAL
