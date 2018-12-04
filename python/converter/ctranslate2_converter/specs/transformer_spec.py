"""Declares specification of the Transformer model."""

from ctranslate2_converter.specs import model_spec


class TransformerSpec(model_spec.LayerSpec):
    def __init__(self, num_layers, fused_projections=False):
        self.encoder = TransformerEncoderSpec(num_layers, fused_projections=fused_projections)
        self.decoder = TransformerDecoderSpec(num_layers, fused_projections=fused_projections)

class TransformerEncoderSpec(model_spec.LayerSpec):
    def __init__(self, num_layers, fused_projections=False):
        self.embeddings = EmbeddingsSpec()
        self.position_encodings = PositionEncoderSpec()
        self.layer_norm = LayerNormSpec()
        self.layer = [
            TransformerEncoderLayerSpec(fused_projections=fused_projections)
            for _ in range(num_layers)]

class TransformerDecoderSpec(model_spec.LayerSpec):
    def __init__(self, num_layers, fused_projections=False):
        self.embeddings = EmbeddingsSpec()
        self.position_encodings = PositionEncoderSpec()
        self.layer_norm = LayerNormSpec()
        self.projection = LinearSpec()
        self.layer = [
            TransformerDecoderLayerSpec(fused_projections=fused_projections)
            for _ in range(num_layers)]

class TransformerEncoderLayerSpec(model_spec.LayerSpec):
    def __init__(self, fused_projections=False):
        self.self_attention = MultiHeadAttentionSpec(
            fused_projections=fused_projections, self_attention=True)
        self.ffn = FeedForwardSpec()

class TransformerDecoderLayerSpec(model_spec.LayerSpec):
    def __init__(self, fused_projections=False):
        self.self_attention = MultiHeadAttentionSpec(
            fused_projections=fused_projections, self_attention=True)
        self.attention = MultiHeadAttentionSpec(
            fused_projections=fused_projections)
        self.ffn = FeedForwardSpec()

class MultiHeadAttentionSpec(model_spec.LayerSpec):
    def __init__(self, fused_projections=False, self_attention=False):
        self.layer_norm = LayerNormSpec()
        if fused_projections:
            if self_attention:
                num_projections = 2
            else:
                num_projections = 3
        else:
            num_projections = 4
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
        self.bias = None

class EmbeddingsSpec(model_spec.LayerSpec):
    def __init__(self):
        self.weight = None

class PositionEncoderSpec(model_spec.LayerSpec):
    def __init__(self):
        self.encodings = model_spec.OPTIONAL
