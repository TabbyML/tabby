"""Declares specification of the Transformer model."""

import numpy as np

from ctranslate2.specs import attention_spec
from ctranslate2.specs import common_spec
from ctranslate2.specs import model_spec


class TransformerSpec(model_spec.ModelSpec):
    """Describes a Transformer model.

    The specification is invariant to hidden dimensions but requires to
    explicitly set the number of layers and attention heads.
    """
    def __init__(self, num_layers, num_heads, with_relative_position=False):
        if isinstance(num_layers, (list, tuple)):
            num_encoder_layers, num_decoder_layers = num_layers
        else:
            num_encoder_layers, num_decoder_layers = num_layers, num_layers
        self.num_heads = np.dtype("int8").type(num_heads)
        self.with_relative_position = with_relative_position
        self.encoder = TransformerEncoderSpec(num_encoder_layers)
        self.decoder = TransformerDecoderSpec(num_decoder_layers)

    @property
    def name(self):
        return "TransformerSpec"

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
        self.embeddings = common_spec.EmbeddingsSpec()
        self.position_encodings = PositionEncoderSpec()
        self.layer_norm = common_spec.LayerNormSpec()
        self.layer = [TransformerEncoderLayerSpec() for _ in range(num_layers)]

class TransformerDecoderSpec(model_spec.LayerSpec):
    def __init__(self, num_layers):
        self.embeddings = common_spec.EmbeddingsSpec()
        self.position_encodings = PositionEncoderSpec()
        self.layer_norm = common_spec.LayerNormSpec()
        self.projection = common_spec.LinearSpec()
        self.layer = [
            TransformerDecoderLayerSpec() for _ in range(num_layers)]

class TransformerEncoderLayerSpec(model_spec.LayerSpec):
    def __init__(self):
        self.self_attention = attention_spec.MultiHeadAttentionSpec(self_attention=True)
        self.ffn = FeedForwardSpec()

class TransformerDecoderLayerSpec(model_spec.LayerSpec):
    def __init__(self):
        self.self_attention = attention_spec.MultiHeadAttentionSpec(self_attention=True)
        self.attention = attention_spec.MultiHeadAttentionSpec()
        self.ffn = FeedForwardSpec()

class FeedForwardSpec(model_spec.LayerSpec):
    def __init__(self):
        self.layer_norm = common_spec.LayerNormSpec()
        self.linear_0 = common_spec.LinearSpec()
        self.linear_1 = common_spec.LinearSpec()

class PositionEncoderSpec(model_spec.LayerSpec):
    def __init__(self):
        self.encodings = model_spec.OPTIONAL
