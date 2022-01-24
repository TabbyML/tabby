"""Declares specification of the Transformer model."""

import numpy as np

from ctranslate2.specs import attention_spec
from ctranslate2.specs import common_spec
from ctranslate2.specs import model_spec


class TransformerSpec(model_spec.SequenceToSequenceModelSpec):
    """Describes a Transformer model.

    The specification is invariant to hidden dimensions but requires to
    explicitly set the number of layers and attention heads.
    """

    def __init__(
        self,
        num_layers,
        num_heads,
        with_relative_position=False,
        pre_norm=True,
        activation=common_spec.Activation.RELU,
        alignment_layer=-1,
        alignment_heads=1,
        num_source_embeddings=1,
        embeddings_merge=common_spec.EmbeddingsMerge.CONCAT,
    ):
        """Initializes a Transformer model specification.

        Args:
          num_layers: Number of encoder and decoder layers, or a 2-tuple if the
            number is different.
          num_heads: Number of attention heads.
          with_relative_position: Enable relative position representations modules.
          pre_norm: Enable the pre-norm Transformer architecture.
          activation: Activation to apply in the feed-forward network.
          alignment_layer: Layer index selected for alignment.
          alignment_heads: Number of attention heads selected for alignment.
          num_source_embeddings: Number of source embeddings.
          embeddings_merge: When num_source_embeddings > 1, specify how the
            embeddings are merged.
        """
        if isinstance(num_layers, (list, tuple)):
            num_encoder_layers, num_decoder_layers = num_layers
        else:
            num_encoder_layers, num_decoder_layers = num_layers, num_layers
        self.num_heads = np.dtype("int8").type(num_heads)
        self.pre_norm = np.dtype("int8").type(pre_norm)
        self.activation = np.dtype("int8").type(activation)
        self.alignment_layer = np.dtype("int16").type(alignment_layer)
        self.alignment_heads = np.dtype("int16").type(alignment_heads)
        self.with_relative_position = with_relative_position
        self.embeddings_merge = np.dtype("int8").type(embeddings_merge)
        self.encoder = TransformerEncoderSpec(
            num_encoder_layers,
            pre_norm=pre_norm,
            num_source_embeddings=num_source_embeddings,
        )
        self.decoder = TransformerDecoderSpec(num_decoder_layers, pre_norm=pre_norm)
        super().__init__(
            source_embeddings_specs=self.encoder.embeddings,
            target_embeddings_specs=[self.decoder.embeddings],
        )

    @property
    def name(self):
        return "TransformerSpec"

    @property
    def revision(self):
        return 4


class TransformerEncoderSpec(model_spec.LayerSpec):
    def __init__(self, num_layers, pre_norm=True, num_source_embeddings=1):
        self.embeddings = [
            common_spec.EmbeddingsSpec() for _ in range(num_source_embeddings)
        ]
        self.scale_embeddings = True
        self.position_encodings = PositionEncoderSpec()
        self.layer_norm = (
            common_spec.LayerNormSpec() if pre_norm else model_spec.OPTIONAL
        )
        self.layer = [TransformerEncoderLayerSpec() for _ in range(num_layers)]


class TransformerDecoderSpec(model_spec.LayerSpec):
    def __init__(self, num_layers, pre_norm=True):
        self.embeddings = common_spec.EmbeddingsSpec()
        self.scale_embeddings = True
        self.position_encodings = PositionEncoderSpec()
        self.layer_norm = (
            common_spec.LayerNormSpec() if pre_norm else model_spec.OPTIONAL
        )
        self.projection = common_spec.LinearSpec()
        self.layer = [TransformerDecoderLayerSpec() for _ in range(num_layers)]


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
