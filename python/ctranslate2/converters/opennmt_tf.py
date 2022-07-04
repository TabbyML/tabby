import argparse
import copy
import os
import warnings

from typing import Dict, List, Optional, Union

import numpy as np

from ctranslate2.converters import utils
from ctranslate2.converters.converter import Converter
from ctranslate2.specs import common_spec, transformer_spec
from ctranslate2.specs.model_spec import OPTIONAL, ModelSpec

_SUPPORTED_ACTIVATIONS = {
    "gelu": common_spec.Activation.GELU,
    "relu": common_spec.Activation.RELU,
    "swish": common_spec.Activation.SWISH,
}


class OpenNMTTFConverterV2(Converter):
    """Converts OpenNMT-tf models."""

    @classmethod
    def from_config(
        cls,
        config: Union[str, dict],
        auto_config: bool = False,
        checkpoint_path: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Creates the converter from the configuration.

        Arguments:
          config: Path to the YAML configuration, or a dictionary with the loaded configuration.
          auto_config: Whether the model automatic configuration values should be used.
          checkpoint_path: Path to the checkpoint or checkpoint directory to load. If not set,
            the latest checkpoint from the model directory is loaded.
          model: If the model instance cannot be resolved from the model directory, this argument
             can be set to either the name of the model in the catalog or the path to the model
             configuration.

        Returns:
          A :class:`ctranslate2.converters.OpenNMTTFConverterV2` instance.
        """
        from opennmt import config as config_util
        from opennmt.utils.checkpoint import Checkpoint

        if isinstance(config, str):
            config = config_util.load_config([config])
        else:
            config = copy.deepcopy(config)

        if model is None:
            model = config_util.load_model(config["model_dir"])
        elif os.path.exists(model):
            model = config_util.load_model_from_file(model)
        else:
            model = config_util.load_model_from_catalog(model)

        if auto_config:
            config_util.merge_config(config, model.auto_config())

        data_config = config_util.try_prefix_paths(config["model_dir"], config["data"])
        model.initialize(data_config)

        checkpoint = Checkpoint.from_config(config, model)
        checkpoint_path = checkpoint.restore(checkpoint_path=checkpoint_path)
        if checkpoint_path is None:
            raise RuntimeError("No checkpoint was restored")

        model.create_variables()
        return cls(model)

    def __init__(self, model):
        """Initializes the converter.

        Arguments:
          model: An initialized and fully-built ``opennmt.models.Model`` instance.
        """
        self._model = model

    def _load(self):
        import opennmt

        if isinstance(self._model, opennmt.models.LanguageModel):
            spec_builder = TransformerDecoderSpecBuilder()
        else:
            spec_builder = TransformerSpecBuilder()

        return spec_builder(self._model)


class TransformerSpecBuilder:
    def __call__(self, model):
        import opennmt

        check = utils.ConfigurationChecker()
        check(
            isinstance(model, opennmt.models.Transformer),
            "Only Transformer models are supported",
        )
        check.validate()

        check(
            isinstance(model.encoder, opennmt.encoders.SelfAttentionEncoder),
            "Parallel encoders are not supported",
        )
        check(
            isinstance(
                model.features_inputter,
                (opennmt.inputters.WordEmbedder, opennmt.inputters.ParallelInputter),
            ),
            "Source inputter must be a WordEmbedder or a ParallelInputter",
        )
        check.validate()

        mha = model.encoder.layers[0].self_attention.layer
        ffn = model.encoder.layers[0].ffn.layer
        with_relative_position = mha.maximum_relative_position is not None
        activation_name = ffn.inner.activation.__name__

        check(
            activation_name in _SUPPORTED_ACTIVATIONS,
            "Activation %s is not supported (supported activations are: %s)"
            % (activation_name, ", ".join(_SUPPORTED_ACTIVATIONS.keys())),
        )
        check(
            with_relative_position != bool(model.encoder.position_encoder),
            "Relative position representation and position encoding cannot be both enabled "
            "or both disabled",
        )
        check(
            model.decoder.attention_reduction
            != opennmt.layers.MultiHeadAttentionReduction.AVERAGE_ALL_LAYERS,
            "Averaging all multi-head attention matrices is not supported",
        )

        source_inputters = _get_inputters(model.features_inputter)
        target_inputters = _get_inputters(model.labels_inputter)
        num_source_embeddings = len(source_inputters)
        if num_source_embeddings == 1:
            embeddings_merge = common_spec.EmbeddingsMerge.CONCAT
        else:
            reducer = model.features_inputter.reducer
            embeddings_merge = None
            if reducer is not None:
                if isinstance(reducer, opennmt.layers.ConcatReducer):
                    embeddings_merge = common_spec.EmbeddingsMerge.CONCAT
                elif isinstance(reducer, opennmt.layers.SumReducer):
                    embeddings_merge = common_spec.EmbeddingsMerge.ADD

            check(
                all(
                    isinstance(inputter, opennmt.inputters.WordEmbedder)
                    for inputter in source_inputters
                ),
                "All source inputters must WordEmbedders",
            )
            check(
                embeddings_merge is not None,
                "Unsupported embeddings reducer %s" % reducer,
            )

        alignment_layer = -1
        alignment_heads = 1
        if (
            model.decoder.attention_reduction
            == opennmt.layers.MultiHeadAttentionReduction.AVERAGE_LAST_LAYER
        ):
            alignment_heads = 0

        check.validate()

        spec = transformer_spec.TransformerSpec(
            (len(model.encoder.layers), len(model.decoder.layers)),
            mha.num_heads,
            with_relative_position=with_relative_position,
            pre_norm=model.encoder.layer_norm is not None,
            activation=_SUPPORTED_ACTIVATIONS[activation_name],
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            num_source_embeddings=num_source_embeddings,
            embeddings_merge=embeddings_merge,
        )

        spec.with_source_bos = bool(source_inputters[0].mark_start)
        spec.with_source_eos = bool(source_inputters[0].mark_end)
        for inputter in source_inputters:
            spec.register_source_vocabulary(_load_vocab(inputter.vocabulary_file))
        for inputter in target_inputters:
            spec.register_target_vocabulary(_load_vocab(inputter.vocabulary_file))

        self.set_transformer_encoder(
            spec.encoder,
            model.encoder,
            model.features_inputter,
        )
        self.set_transformer_decoder(
            spec.decoder,
            model.decoder,
            model.labels_inputter,
        )

        return spec

    def set_transformer_encoder(self, spec, module, inputter):
        for embedding_spec, inputter in zip(spec.embeddings, _get_inputters(inputter)):
            self.set_embeddings(embedding_spec, inputter)
        if module.position_encoder is not None:
            self.set_position_encodings(
                spec.position_encodings,
                module.position_encoder,
            )

        for layer_spec, layer in zip(spec.layer, module.layers):
            self.set_multi_head_attention(
                layer_spec.self_attention,
                layer.self_attention,
                self_attention=True,
            )

            self.set_ffn(layer_spec.ffn, layer.ffn)

        if module.layer_norm is not None:
            self.set_layer_norm(spec.layer_norm, module.layer_norm)

    def set_transformer_decoder(self, spec, module, inputter):
        self.set_embeddings(spec.embeddings, inputter)
        if module.position_encoder is not None:
            self.set_position_encodings(
                spec.position_encodings,
                module.position_encoder,
            )

        for layer_spec, layer in zip(spec.layer, module.layers):
            self.set_multi_head_attention(
                layer_spec.self_attention,
                layer.self_attention,
                self_attention=True,
            )

            if layer.attention:
                self.set_multi_head_attention(
                    layer_spec.attention,
                    layer.attention[0],
                    self_attention=False,
                )

            self.set_ffn(layer_spec.ffn, layer.ffn)

        if module.layer_norm is not None:
            self.set_layer_norm(spec.layer_norm, module.layer_norm)

        self.set_linear(spec.projection, module.output_layer)

    def set_ffn(self, spec, module):
        self.set_linear(spec.linear_0, module.layer.inner)
        self.set_linear(spec.linear_1, module.layer.outer)
        self.set_layer_norm_from_wrapper(spec.layer_norm, module)

    def set_multi_head_attention(self, spec, module, self_attention=False):
        split_layers = [common_spec.LinearSpec() for _ in range(3)]
        self.set_linear(split_layers[0], module.layer.linear_queries)
        self.set_linear(split_layers[1], module.layer.linear_keys)
        self.set_linear(split_layers[2], module.layer.linear_values)

        if self_attention:
            utils.fuse_linear(spec.linear[0], split_layers)
            if module.layer.maximum_relative_position is not None:
                spec.relative_position_keys = (
                    module.layer.relative_position_keys.numpy()
                )
                spec.relative_position_values = (
                    module.layer.relative_position_values.numpy()
                )
        else:
            utils.fuse_linear(spec.linear[0], split_layers[:1])
            utils.fuse_linear(spec.linear[1], split_layers[1:])

        self.set_linear(spec.linear[-1], module.layer.linear_output)
        self.set_layer_norm_from_wrapper(spec.layer_norm, module)

    def set_layer_norm_from_wrapper(self, spec, module):
        self.set_layer_norm(
            spec,
            module.output_layer_norm
            if module.input_layer_norm is None
            else module.input_layer_norm,
        )

    def set_layer_norm(self, spec, module):
        spec.gamma = module.gamma.numpy()
        spec.beta = module.beta.numpy()

    def set_linear(self, spec, module):
        spec.weight = module.kernel.numpy()
        if not module.transpose:
            spec.weight = spec.weight.transpose()
        if module.bias is not None:
            spec.bias = module.bias.numpy()

    def set_embeddings(self, spec, module):
        spec.weight = module.embedding.numpy()

    def set_position_encodings(self, spec, module):
        import opennmt

        if isinstance(module, opennmt.layers.PositionEmbedder):
            spec.encodings = module.embedding.numpy()[1:]


class TransformerDecoderSpecBuilder(TransformerSpecBuilder):
    def __call__(self, model):
        import opennmt

        check = utils.ConfigurationChecker()
        check(
            isinstance(model.decoder, opennmt.decoders.SelfAttentionDecoder),
            "Only self-attention decoders are supported",
        )
        check.validate()

        mha = model.decoder.layers[0].self_attention.layer
        ffn = model.decoder.layers[0].ffn.layer
        activation_name = ffn.inner.activation.__name__

        check(
            activation_name in _SUPPORTED_ACTIVATIONS,
            "Activation %s is not supported (supported activations are: %s)"
            % (activation_name, ", ".join(_SUPPORTED_ACTIVATIONS.keys())),
        )
        check.validate()

        spec = transformer_spec.TransformerDecoderModelSpec(
            len(model.decoder.layers),
            mha.num_heads,
            pre_norm=model.decoder.layer_norm is not None,
            activation=_SUPPORTED_ACTIVATIONS[activation_name],
        )

        spec.register_vocabulary(_load_vocab(model.features_inputter.vocabulary_file))
        self.set_transformer_decoder(
            spec.decoder,
            model.decoder,
            model.features_inputter,
        )

        return spec


def _get_inputters(inputter):
    import opennmt

    return (
        inputter.inputters
        if isinstance(inputter, opennmt.inputters.MultiInputter)
        else [inputter]
    )


def load_model(model_path):
    """Loads variables from a TensorFlow checkpoint."""
    import tensorflow as tf

    if tf.saved_model.contains_saved_model(model_path):
        raise RuntimeError(
            "Converting the SavedModel format is not supported, "
            "please convert a TensorFlow checkpoint instead"
        )

    if os.path.isdir(model_path):
        checkpoint = tf.train.latest_checkpoint(model_path)
        if checkpoint is None:
            raise ValueError(
                "Checkpoint not found in directory %s. You may need to include "
                "the checkpoint prefix in the path (e.g. model_dir/ckpt-100000)."
                % model_path
            )
    else:
        checkpoint = model_path

    reader = tf.train.load_checkpoint(checkpoint)
    variables = {
        name: reader.get_tensor(name)
        for name in reader.get_variable_to_shape_map().keys()
    }

    model_version = 1
    if os.path.basename(checkpoint).startswith("ckpt"):
        model_version = 2
        variables = {
            name.replace("/.ATTRIBUTES/VARIABLE_VALUE", ""): value
            for name, value in variables.items()
        }

    return model_version, variables


def _load_vocab(vocab, unk_token="<unk>"):
    import opennmt

    if isinstance(vocab, opennmt.data.Vocab):
        tokens = list(vocab.words)
    elif isinstance(vocab, list):
        tokens = list(vocab)
    elif isinstance(vocab, str):
        tokens = opennmt.data.Vocab.from_file(vocab).words
    else:
        raise TypeError("Invalid vocabulary type")

    if unk_token not in tokens:
        tokens.append(unk_token)
    return tokens


class OpenNMTTFConverter(Converter):
    """Converts OpenNMT-tf models (deprecated).

    Warning:
      This converter is deprecated because it has limited support for custom models and
      configurations. Please use the new :class:`ctranslate2.converters.OpenNMTTFConverterV2`
      instead.
    """

    def __init__(
        self,
        model_spec: ModelSpec,
        src_vocab: Union[str, List[str]],
        tgt_vocab: Union[str, List[str]],
        model_path: Optional[str] = None,
        variables: Optional[Dict[str, np.ndarray]] = None,
    ):
        """Initializes the OpenNMT-tf converter.

        Arguments:
          model_spec: Specification of the model to convert.
          src_vocab: Path to the source vocabulary or list of tokens.
          tgt_vocab: Path to the target vocabulary or list of tokens.
          model_path: Path to the OpenNMT-tf checkpoint
            (mutually exclusive with :obj:`variables`).
          variables: Dictionary of variables name to value
            (mutually exclusive with :obj:`model_path`).

        Raises:
          ValueError: If none or both of :obj:`model_path` and :obj:`variables` are set.
        """
        warnings.warn(
            "OpenNMTTFConverter is deprecated because it has limited support for "
            "custom models and configurations. Please use the new OpenNMTTFConverterV2 instead.",
            DeprecationWarning,
        )

        if (model_path is None) == (variables is None):
            raise ValueError("Exactly one of model_path and variables should be set")
        if variables is not None and not isinstance(variables, dict):
            raise ValueError(
                "variables should be a dict mapping variable name to value"
            )
        self._model_spec = model_spec
        self._model_path = model_path
        self._src_vocab = src_vocab
        self._tgt_vocab = tgt_vocab
        self._variables = variables

    def _load(self):
        model_spec = self._model_spec
        if self._model_path is not None:
            version, variables = load_model(self._model_path)
        else:
            version = 2  # Assume we are passing V2 variables.
            variables = self._variables
        if version >= 2:
            set_transformer_spec_v2(model_spec, variables)
        else:
            set_transformer_spec(model_spec, variables)
        model_spec.register_source_vocabulary(_load_vocab(self._src_vocab))
        model_spec.register_target_vocabulary(_load_vocab(self._tgt_vocab))
        return model_spec


def set_transformer_spec_v2(spec, variables):
    set_embeddings(
        spec.encoder.embeddings[0],
        variables,
        "model/examples_inputter/features_inputter",
        version=2,
    )
    try:
        target_embedding_name = set_embeddings(
            spec.decoder.embeddings,
            variables,
            "model/examples_inputter/labels_inputter",
            version=2,
        )
    except KeyError:
        target_embedding_name = set_embeddings(
            spec.decoder.embeddings,
            variables,
            "model/examples_inputter/features_inputter",
            version=2,
        )
    set_transformer_encoder_v2(spec.encoder, variables, "model/encoder")
    set_transformer_decoder_v2(
        spec.decoder,
        variables,
        "model/decoder",
        target_embedding_name,
    )


def set_transformer_encoder_v2(spec, variables, scope):
    if spec.layer_norm != OPTIONAL:
        set_layer_norm(spec.layer_norm, variables, "%s/layer_norm" % scope)
    for i, layer in enumerate(spec.layer):
        set_transformer_encoder_layer_v2(layer, variables, "%s/layers/%d" % (scope, i))


def set_transformer_decoder_v2(spec, variables, scope, target_embedding_name):
    try:
        set_linear(
            spec.projection,
            variables,
            "%s/output_layer" % scope,
            transpose=False,
        )
        if not np.array_equal(spec.projection.weight, spec.embeddings.weight):
            spec.projection.weight = spec.projection.weight.transpose()
    except KeyError:
        set_linear(
            spec.projection,
            variables,
            "%s/output_layer" % scope,
            weight_name=target_embedding_name,
            transpose=False,
        )
    if spec.layer_norm != OPTIONAL:
        set_layer_norm(spec.layer_norm, variables, "%s/layer_norm" % scope)
    for i, layer in enumerate(spec.layer):
        set_transformer_decoder_layer_v2(layer, variables, "%s/layers/%d" % (scope, i))


def set_transformer_encoder_layer_v2(spec, variables, scope):
    set_ffn_v2(spec.ffn, variables, "%s/ffn" % scope)
    set_multi_head_attention_v2(
        spec.self_attention,
        variables,
        "%s/self_attention" % scope,
        self_attention=True,
    )


def set_transformer_decoder_layer_v2(spec, variables, scope):
    set_ffn_v2(spec.ffn, variables, "%s/ffn" % scope)
    set_multi_head_attention_v2(
        spec.self_attention,
        variables,
        "%s/self_attention" % scope,
        self_attention=True,
    )
    set_multi_head_attention_v2(spec.attention, variables, "%s/attention/0" % scope)


def set_ffn_v2(spec, variables, scope):
    try:
        set_layer_norm(spec.layer_norm, variables, "%s/input_layer_norm" % scope)
    except KeyError:
        set_layer_norm(spec.layer_norm, variables, "%s/output_layer_norm" % scope)
    set_linear(spec.linear_0, variables, "%s/layer/inner" % scope)
    set_linear(spec.linear_1, variables, "%s/layer/outer" % scope)


def set_multi_head_attention_v2(spec, variables, scope, self_attention=False):
    try:
        set_layer_norm(spec.layer_norm, variables, "%s/input_layer_norm" % scope)
    except KeyError:
        set_layer_norm(spec.layer_norm, variables, "%s/output_layer_norm" % scope)
    if self_attention:
        split_layers = [common_spec.LinearSpec() for _ in range(3)]
        set_linear(split_layers[0], variables, "%s/layer/linear_queries" % scope)
        set_linear(split_layers[1], variables, "%s/layer/linear_keys" % scope)
        set_linear(split_layers[2], variables, "%s/layer/linear_values" % scope)
        utils.fuse_linear(spec.linear[0], split_layers)
        if spec.relative_position_keys is None:
            spec.relative_position_keys = variables[
                "%s/layer/relative_position_keys" % scope
            ]
            spec.relative_position_values = variables[
                "%s/layer/relative_position_values" % scope
            ]
    else:
        set_linear(spec.linear[0], variables, "%s/layer/linear_queries" % scope)
        split_layers = [common_spec.LinearSpec() for _ in range(2)]
        set_linear(split_layers[0], variables, "%s/layer/linear_keys" % scope)
        set_linear(split_layers[1], variables, "%s/layer/linear_values" % scope)
        utils.fuse_linear(spec.linear[1], split_layers)
    set_linear(spec.linear[-1], variables, "%s/layer/linear_output" % scope)


def set_transformer_spec(spec, variables):
    if spec.with_relative_position:
        raise NotImplementedError()
    set_transformer_encoder(spec.encoder, variables)
    set_transformer_decoder(spec.decoder, variables)


def set_transformer_encoder(spec, variables):
    set_layer_norm(spec.layer_norm, variables, "transformer/encoder/LayerNorm")
    try:
        set_embeddings(spec.embeddings[0], variables, "transformer/encoder")
    except KeyError:
        # Try shared embeddings scope instead.
        set_embeddings(spec.embeddings[0], variables, "transformer/shared_embeddings")
    for i, layer in enumerate(spec.layer):
        set_transformer_encoder_layer(
            layer, variables, "transformer/encoder/layer_%d" % i
        )


def set_transformer_decoder(spec, variables):
    try:
        embeddings_name = set_embeddings(
            spec.embeddings, variables, "transformer/decoder"
        )
    except KeyError:
        # Try shared embeddings scope instead.
        embeddings_name = set_embeddings(
            spec.embeddings, variables, "transformer/shared_embeddings"
        )
    try:
        set_linear(spec.projection, variables, "transformer/decoder/dense")
    except KeyError:
        # Try reusing the target embeddings.
        set_linear(
            spec.projection,
            variables,
            "transformer",
            weight_name=embeddings_name,
            transpose=False,
        )
    set_layer_norm(spec.layer_norm, variables, "transformer/decoder/LayerNorm")
    for i, layer in enumerate(spec.layer):
        set_transformer_decoder_layer(
            layer, variables, "transformer/decoder/layer_%d" % i
        )


def set_transformer_encoder_layer(spec, variables, scope):
    set_ffn(spec.ffn, variables, "%s/ffn" % scope)
    set_multi_head_attention(
        spec.self_attention, variables, "%s/multi_head" % scope, self_attention=True
    )


def set_transformer_decoder_layer(spec, variables, scope):
    set_ffn(spec.ffn, variables, "%s/ffn" % scope)
    set_multi_head_attention(
        spec.self_attention,
        variables,
        "%s/masked_multi_head" % scope,
        self_attention=True,
    )
    set_multi_head_attention(spec.attention, variables, "%s/multi_head" % scope)


def set_ffn(spec, variables, scope):
    set_layer_norm(spec.layer_norm, variables, "%s/LayerNorm" % scope)
    set_linear(spec.linear_0, variables, "%s/conv1d" % scope)
    set_linear(spec.linear_1, variables, "%s/conv1d_1" % scope)


def set_multi_head_attention(spec, variables, scope, self_attention=False):
    set_layer_norm(spec.layer_norm, variables, "%s/LayerNorm" % scope)
    set_linear(spec.linear[0], variables, "%s/conv1d" % scope)
    set_linear(spec.linear[1], variables, "%s/conv1d_1" % scope)
    if not self_attention:
        set_linear(spec.linear[2], variables, "%s/conv1d_2" % scope)


def set_layer_norm(spec, variables, scope):
    spec.gamma = variables["%s/gamma" % scope]
    spec.beta = variables["%s/beta" % scope]


def set_linear(spec, variables, scope, weight_name=None, transpose=True):
    if weight_name is None:
        weight_name = "%s/kernel" % scope
    spec.weight = variables[weight_name].squeeze()
    if transpose:
        spec.weight = spec.weight.transpose()
    bias = variables.get("%s/bias" % scope)
    if bias is not None:
        spec.bias = bias


def set_embeddings(spec, variables, scope, version=1):
    if version == 2:
        name = "embedding"
    else:
        name = "w_embs"
    variable_name = "%s/%s" % (scope, name)
    spec.weight = variables[variable_name]
    return variable_name


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", help="Path to the YAML configuration.")
    parser.add_argument(
        "--auto_config",
        action="store_true",
        help="Use the model automatic configuration values.",
    )
    parser.add_argument(
        "--model_path",
        help=(
            "Path to the checkpoint or checkpoint directory to load. If not set, "
            "the latest checkpoint from the model directory is loaded."
        ),
    )
    parser.add_argument(
        "--model_type",
        help=(
            "If the model instance cannot be resolved from the model directory, "
            "this argument can be set to either the name of the model in the catalog "
            "or the path to the model configuration."
        ),
    )
    parser.add_argument(
        "--src_vocab",
        help="Path to the source vocabulary (required if no configuration is set).",
    )
    parser.add_argument(
        "--tgt_vocab",
        help="Path to the target vocabulary (required if no configuration is set).",
    )
    Converter.declare_arguments(parser)
    args = parser.parse_args()

    config = args.config
    if not config:
        if not args.model_path or not args.src_vocab or not args.tgt_vocab:
            raise ValueError(
                "Options --model_path, --src_vocab, --tgt_vocab are required "
                "when a configuration is not set"
            )

        model_dir = (
            args.model_path
            if os.path.isdir(args.model_path)
            else os.path.dirname(args.model_path)
        )
        config = {
            "model_dir": model_dir,
            "data": {
                "source_vocabulary": args.src_vocab,
                "target_vocabulary": args.tgt_vocab,
            },
        }

    converter = OpenNMTTFConverterV2.from_config(
        config,
        auto_config=args.auto_config,
        checkpoint_path=args.model_path,
        model=args.model_type,
    )
    converter.convert_from_args(args)


if __name__ == "__main__":
    main()
