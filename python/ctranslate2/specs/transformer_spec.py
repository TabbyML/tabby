"""Declares specification of the Transformer model."""

from typing import Tuple, Union

import numpy as np

from ctranslate2.specs import attention_spec, common_spec, model_spec


class TransformerSpec(model_spec.SequenceToSequenceModelSpec):
    """Describes a Transformer model.

    The specification is invariant to hidden dimensions but requires to
    explicitly set the number of layers and attention heads.
    """

    def __init__(
        self,
        num_layers: Union[int, Tuple[int]],
        num_heads: int,
        with_relative_position: bool = False,
        pre_norm: bool = True,
        activation: common_spec.Activation = common_spec.Activation.RELU,
        alignment_layer: int = -1,
        alignment_heads: int = 1,
        num_source_embeddings: int = 1,
        embeddings_merge: common_spec.EmbeddingsMerge = common_spec.EmbeddingsMerge.CONCAT,
        layernorm_embedding: bool = False,
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
          embeddings_merge: When :obj:`num_source_embeddings` > 1, specify how the
            embeddings are merged.
          layernorm_embedding: Apply layer normalization after the embedding layer.
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
        self.layernorm_embedding = np.dtype("int8").type(layernorm_embedding)
        self.encoder = TransformerEncoderSpec(
            num_encoder_layers,
            pre_norm=pre_norm,
            num_source_embeddings=num_source_embeddings,
            layernorm_embedding=layernorm_embedding,
        )
        self.decoder = TransformerDecoderSpec(
            num_decoder_layers,
            pre_norm=pre_norm,
            layernorm_embedding=layernorm_embedding,
        )
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


class TransformerDecoderModelSpec(model_spec.LanguageModelSpec):
    """Describes a Transformer decoder model (e.g. GPT-2)."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        pre_norm: bool = True,
        activation: common_spec.Activation = common_spec.Activation.RELU,
        layernorm_embedding: bool = False,
        no_final_norm: bool = False,
        project_in_out: bool = False,
    ):
        """Initializes a Transformer decoder model specification.

        Args:
          num_layers: Number of decoder layers.
          num_heads: Number of attention heads.
          pre_norm: Enable the pre-norm Transformer architecture.
          activation: Activation to apply in the feed-forward network.
          layernorm_embedding: Apply layer normalization after the embedding layer.
          no_final_norm: Do not apply layer normalization after the last decoder block.
          project_in_out: Add a linear layer after the embedding layer and another one
            before the final output projection.
        """
        self.num_heads = np.dtype("int16").type(num_heads)
        self.pre_norm = pre_norm
        self.activation = np.dtype("int8").type(activation)
        self.layernorm_embedding = layernorm_embedding
        self.no_final_norm = no_final_norm
        self.project_in_out = project_in_out
        self.decoder = TransformerDecoderSpec(
            num_layers,
            pre_norm=pre_norm,
            layernorm_embedding=layernorm_embedding,
            with_encoder_attention=False,
            no_final_norm=no_final_norm,
            project_in_out=project_in_out,
        )
        super().__init__(self.decoder.embeddings)

    @property
    def name(self):
        return "TransformerDecoderSpec"

    @property
    def revision(self):
        return 1


class TransformerEncoderSpec(model_spec.LayerSpec):
    def __init__(
        self,
        num_layers,
        pre_norm=True,
        num_source_embeddings=1,
        layernorm_embedding=False,
    ):
        self.embeddings = [
            common_spec.EmbeddingsSpec() for _ in range(num_source_embeddings)
        ]
        self.scale_embeddings = True
        self.position_encodings = PositionEncoderSpec()
        self.layer_norm = (
            common_spec.LayerNormSpec() if pre_norm else model_spec.OPTIONAL
        )
        self.layernorm_embedding = (
            common_spec.LayerNormSpec() if layernorm_embedding else model_spec.OPTIONAL
        )
        self.layer = [TransformerEncoderLayerSpec() for _ in range(num_layers)]


class TransformerDecoderSpec(model_spec.LayerSpec):
    def __init__(
        self,
        num_layers,
        pre_norm=True,
        layernorm_embedding=False,
        with_encoder_attention=True,
        no_final_norm=False,
        project_in_out=False,
    ):
        self.embeddings = common_spec.EmbeddingsSpec()
        self.scale_embeddings = True
        self.position_encodings = PositionEncoderSpec()
        self.layer_norm = (
            common_spec.LayerNormSpec()
            if pre_norm and not no_final_norm
            else model_spec.OPTIONAL
        )
        self.layernorm_embedding = (
            common_spec.LayerNormSpec() if layernorm_embedding else model_spec.OPTIONAL
        )
        self.projection = common_spec.LinearSpec()
        self.layer = [
            TransformerDecoderLayerSpec(with_encoder_attention=with_encoder_attention)
            for _ in range(num_layers)
        ]
        self.start_from_zero_embedding = False

        if project_in_out:
            self.project_in = common_spec.LinearSpec()
            self.project_out = common_spec.LinearSpec()
        else:
            self.project_in = model_spec.OPTIONAL
            self.project_out = model_spec.OPTIONAL


class TransformerEncoderLayerSpec(model_spec.LayerSpec):
    def __init__(self):
        self.self_attention = attention_spec.MultiHeadAttentionSpec(self_attention=True)
        self.ffn = FeedForwardSpec()


class TransformerDecoderLayerSpec(model_spec.LayerSpec):
    def __init__(self, with_encoder_attention=True):
        self.self_attention = attention_spec.MultiHeadAttentionSpec(self_attention=True)
        self.attention = (
            attention_spec.MultiHeadAttentionSpec()
            if with_encoder_attention
            else model_spec.OPTIONAL
        )
        self.ffn = FeedForwardSpec()


class FeedForwardSpec(model_spec.LayerSpec):
    def __init__(self):
        self.layer_norm = common_spec.LayerNormSpec()
        self.linear_0 = common_spec.LinearSpec()
        self.linear_1 = common_spec.LinearSpec()


class PositionEncoderSpec(model_spec.LayerSpec):
    def __init__(self):
        self.encodings = model_spec.OPTIONAL
