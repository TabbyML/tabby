"""Declares specification of the Transformer model."""

from typing import Tuple, Union

import numpy as np

from ctranslate2.specs import attention_spec, common_spec, model_spec


class TransformerEncoderSpec(model_spec.LayerSpec):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        pre_norm: bool = True,
        no_final_norm: bool = False,
        activation: common_spec.Activation = common_spec.Activation.RELU,
        num_source_embeddings: int = 1,
        embeddings_merge: common_spec.EmbeddingsMerge = common_spec.EmbeddingsMerge.CONCAT,
        layernorm_embedding: bool = False,
        relative_position: bool = False,
        relative_attention_bias: bool = False,
        ffn_glu: bool = False,
        rms_norm: bool = False,
    ):
        """Initializes a Transformer encoder specification.

        Args:
          num_layers: Number of layers.
          num_heads: Number of attention heads.
          pre_norm: Enable the pre-norm Transformer architecture.
          no_final_norm: Disable the final layer norm in the pre-norm architecture.
          activation: Activation to apply in the feed-forward network.
          num_source_embeddings: Number of source embeddings.
          embeddings_merge: When :obj:`num_source_embeddings` > 1, specify how the
            embeddings are merged.
          layernorm_embedding: Apply layer normalization after the embedding layer.
          relative_position: Use relative position representations in the self-attention
            layers as described in https://arxiv.org/abs/1803.02155.
          relative_attention_bias: Use relative attention bias in the self-attention
            layers as described in the T5 paper https://arxiv.org/abs/1910.10683.
          ffn_glu: Use gated linear units in the FFN layers as described in
            https://arxiv.org/abs/2002.05202.
          rms_norm: Use the root mean square layer normalization.
        """
        self.num_heads = np.dtype("int16").type(num_heads)
        self.pre_norm = pre_norm
        self.activation = np.dtype("int8").type(activation)
        self.embeddings_merge = np.dtype("int8").type(embeddings_merge)
        self.embeddings = [
            common_spec.EmbeddingsSpec() for _ in range(num_source_embeddings)
        ]
        self.scale_embeddings = True
        if not relative_position and not relative_attention_bias:
            self.position_encodings = PositionEncoderSpec()
        if pre_norm and not no_final_norm:
            self.layer_norm = common_spec.LayerNormSpec(rms_norm=rms_norm)
        if layernorm_embedding:
            self.layernorm_embedding = common_spec.LayerNormSpec(rms_norm=rms_norm)
        self.layer = [
            TransformerEncoderLayerSpec(
                relative_position=relative_position,
                relative_attention_bias=relative_attention_bias,
                ffn_glu=ffn_glu,
                rms_norm=rms_norm,
            )
            for _ in range(num_layers)
        ]


class TransformerDecoderSpec(model_spec.LayerSpec):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        pre_norm: bool = True,
        activation: common_spec.Activation = common_spec.Activation.RELU,
        layernorm_embedding: bool = False,
        with_encoder_attention: bool = True,
        no_final_norm: bool = False,
        project_in_out: bool = False,
        relative_position: bool = False,
        relative_attention_bias: bool = False,
        alignment_layer: int = -1,
        alignment_heads: int = 1,
        ffn_glu: bool = False,
        rms_norm: bool = False,
        alibi: bool = False,
    ):
        """Initializes a Transformer decoder specification.

        Args:
          num_layers: Number of layers.
          num_heads: Number of attention heads.
          pre_norm: Enable the pre-norm Transformer architecture.
          activation: Activation to apply in the feed-forward network.
          layernorm_embedding: Apply layer normalization after the embedding layer.
          with_encoder_attention: Enable the encoder attention sublayers.
          no_final_norm: Disable the final layer norm in the pre-norm architecture.
          project_in_out: Add linear transformations after the embedding layer and before
            the final layer.
          relative_position: Use relative position representations in the self-attention
            layers as described in https://arxiv.org/abs/1803.02155.
          relative_attention_bias: Use relative attention bias in the self-attention
            layers as described in the T5 paper https://arxiv.org/abs/1910.10683.
          alignment_layer: Layer index selected for alignment.
          alignment_heads: Number of attention heads selected for alignment.
          ffn_glu: Use gated linear units in the FFN layers as described in
            https://arxiv.org/abs/2002.05202.
          rms_norm: Use the root mean square layer normalization.
          alibi: Use attention with linear biases.
        """
        self.num_heads = np.dtype("int16").type(num_heads)
        self.pre_norm = pre_norm
        self.activation = np.dtype("int8").type(activation)
        self.alignment_layer = np.dtype("int16").type(alignment_layer)
        self.alignment_heads = np.dtype("int16").type(alignment_heads)
        self.embeddings = common_spec.EmbeddingsSpec()
        self.scale_embeddings = True
        self.scale_outputs = model_spec.OPTIONAL
        self.alibi = alibi
        if not relative_position and not relative_attention_bias:
            self.position_encodings = PositionEncoderSpec()
        if pre_norm and not no_final_norm:
            self.layer_norm = common_spec.LayerNormSpec(rms_norm=rms_norm)
        if layernorm_embedding:
            self.layernorm_embedding = common_spec.LayerNormSpec(rms_norm=rms_norm)
        self.projection = common_spec.LinearSpec()
        self.layer = [
            TransformerDecoderLayerSpec(
                with_encoder_attention=with_encoder_attention,
                relative_position=relative_position,
                relative_attention_bias=relative_attention_bias,
                ffn_glu=ffn_glu,
                rms_norm=rms_norm,
            )
            for _ in range(num_layers)
        ]
        self.start_from_zero_embedding = False

        if project_in_out:
            self.project_in = common_spec.LinearSpec()
            self.project_out = common_spec.LinearSpec()


class TransformerEncoderLayerSpec(model_spec.LayerSpec):
    def __init__(
        self,
        relative_position=False,
        relative_attention_bias=False,
        ffn_glu=False,
        rms_norm=False,
    ):
        self.self_attention = attention_spec.MultiHeadAttentionSpec(
            self_attention=True,
            relative_position=relative_position,
            relative_attention_bias=relative_attention_bias,
            rms_norm=rms_norm,
        )
        self.ffn = FeedForwardSpec(glu=ffn_glu, rms_norm=rms_norm)


class TransformerDecoderLayerSpec(model_spec.LayerSpec):
    def __init__(
        self,
        with_encoder_attention=True,
        relative_position=False,
        relative_attention_bias=False,
        ffn_glu=False,
        rms_norm=False,
    ):
        self.self_attention = attention_spec.MultiHeadAttentionSpec(
            self_attention=True,
            relative_position=relative_position,
            relative_attention_bias=relative_attention_bias,
            rms_norm=rms_norm,
        )
        if with_encoder_attention:
            self.attention = attention_spec.MultiHeadAttentionSpec(rms_norm=rms_norm)
        self.ffn = FeedForwardSpec(glu=ffn_glu, rms_norm=rms_norm)


class FeedForwardSpec(model_spec.LayerSpec):
    def __init__(self, glu=False, rms_norm=False):
        self.layer_norm = common_spec.LayerNormSpec(rms_norm=rms_norm)
        self.linear_0 = common_spec.LinearSpec()
        self.linear_1 = common_spec.LinearSpec()
        if glu:
            self.linear_0_noact = common_spec.LinearSpec()


class PositionEncoderSpec(model_spec.LayerSpec):
    def __init__(self):
        self.encodings = model_spec.OPTIONAL


class TransformerSpec(model_spec.SequenceToSequenceModelSpec):
    """Describes a Transformer model.

    The specification is invariant to hidden dimensions but requires to
    explicitly set the number of layers and attention heads.
    """

    def __init__(
        self, encoder: TransformerEncoderSpec, decoder: TransformerDecoderSpec
    ):
        """Initializes a Transformer model specification.

        Args:
          encoder: The encoder specification.
          decoder: The decoder specification.
        """
        if not isinstance(encoder, TransformerEncoderSpec):
            raise TypeError("encoder argument must be a TransformerEncoderSpec")
        if not isinstance(decoder, TransformerDecoderSpec):
            raise TypeError("decoder argument must be a TransformerDecoderSpec")

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @classmethod
    def from_config(
        cls,
        num_layers: Union[int, Tuple[int, int]],
        num_heads: int,
        with_relative_position: bool = False,
        pre_norm: bool = True,
        no_final_norm: bool = False,
        activation: common_spec.Activation = common_spec.Activation.RELU,
        alignment_layer: int = -1,
        alignment_heads: int = 1,
        num_source_embeddings: int = 1,
        embeddings_merge: common_spec.EmbeddingsMerge = common_spec.EmbeddingsMerge.CONCAT,
        layernorm_embedding: bool = False,
        relative_attention_bias: bool = False,
        ffn_glu: bool = False,
        rms_norm: bool = False,
    ):
        """Creates a Transformer model specification.

        Args:
          num_layers: Number of encoder and decoder layers, or a 2-tuple if the
            number is different.
          num_heads: Number of attention heads.
          with_relative_position: Use relative position representations in the self-attention
            layers as described in https://arxiv.org/abs/1803.02155.
          pre_norm: Enable the pre-norm Transformer architecture.
          no_final_norm: Disable the final layer norm in the pre-norm architecture.
          activation: Activation to apply in the feed-forward network.
          alignment_layer: Layer index selected for alignment.
          alignment_heads: Number of attention heads selected for alignment.
          num_source_embeddings: Number of source embeddings.
          embeddings_merge: When :obj:`num_source_embeddings` > 1, specify how the
            embeddings are merged.
          layernorm_embedding: Apply layer normalization after the embedding layer.
          relative_attention_bias: Use relative attention bias in the self-attention
            layers as described in the T5 paper https://arxiv.org/abs/1910.10683.
          ffn_glu: Use gated linear units in the FFN layer as described in
            https://arxiv.org/abs/2002.05202.
          rms_norm: Use the root mean square layer normalization.
        """
        if isinstance(num_layers, (list, tuple)):
            num_encoder_layers, num_decoder_layers = num_layers
        else:
            num_encoder_layers, num_decoder_layers = num_layers, num_layers

        encoder = TransformerEncoderSpec(
            num_encoder_layers,
            num_heads,
            pre_norm=pre_norm,
            no_final_norm=no_final_norm,
            activation=activation,
            num_source_embeddings=num_source_embeddings,
            embeddings_merge=embeddings_merge,
            layernorm_embedding=layernorm_embedding,
            relative_position=with_relative_position,
            relative_attention_bias=relative_attention_bias,
            ffn_glu=ffn_glu,
            rms_norm=rms_norm,
        )

        decoder = TransformerDecoderSpec(
            num_decoder_layers,
            num_heads,
            pre_norm=pre_norm,
            no_final_norm=no_final_norm,
            activation=activation,
            layernorm_embedding=layernorm_embedding,
            relative_position=with_relative_position,
            relative_attention_bias=relative_attention_bias,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            ffn_glu=ffn_glu,
            rms_norm=rms_norm,
        )

        return cls(encoder, decoder)

    @property
    def name(self):
        return "TransformerSpec"

    @property
    def revision(self):
        return 7

    def get_source_vocabulary_size(self):
        return [spec.weight.shape[0] for spec in self.encoder.embeddings]

    def get_target_vocabulary_size(self):
        return self.decoder.embeddings.weight.shape[0]


class TransformerDecoderModelSpec(model_spec.LanguageModelSpec):
    """Describes a Transformer decoder model (e.g. GPT-2)."""

    def __init__(self, decoder: TransformerDecoderSpec):
        """Initializes a Transformer decoder model specification.

        Args:
          decoder: The decoder specification.
        """
        if not isinstance(decoder, TransformerDecoderSpec):
            raise TypeError("decoder argument must be a TransformerDecoderSpec")

        super().__init__()
        self.decoder = decoder

    @classmethod
    def from_config(
        cls,
        num_layers: int,
        num_heads: int,
        pre_norm: bool = True,
        activation: common_spec.Activation = common_spec.Activation.RELU,
        layernorm_embedding: bool = False,
        no_final_norm: bool = False,
        project_in_out: bool = False,
        with_relative_position: bool = False,
        alibi: bool = False,
    ):
        """Creates a Transformer decoder model specification.

        Args:
          num_layers: Number of decoder layers.
          num_heads: Number of attention heads.
          pre_norm: Enable the pre-norm Transformer architecture.
          activation: Activation to apply in the feed-forward network.
          layernorm_embedding: Apply layer normalization after the embedding layer.
          no_final_norm: Do not apply layer normalization after the last decoder block.
          project_in_out: Add a linear layer after the embedding layer and another one
            before the final output projection.
          with_relative_position: Enable relative position representations modules.
          alibi: Use attention with linear biases.
        """
        decoder = TransformerDecoderSpec(
            num_layers,
            num_heads,
            pre_norm=pre_norm,
            activation=activation,
            layernorm_embedding=layernorm_embedding,
            with_encoder_attention=False,
            no_final_norm=no_final_norm,
            project_in_out=project_in_out,
            relative_position=with_relative_position,
            alibi=alibi,
        )

        return cls(decoder)

    @property
    def name(self):
        return "TransformerDecoderSpec"

    @property
    def revision(self):
        return 3

    def get_vocabulary_size(self):
        return self.decoder.embeddings.weight.shape[0]
