import shutil
import os

from ctranslate2.converters import utils
from ctranslate2.converters.converter import Converter
from ctranslate2.specs import common_spec
from ctranslate2.specs import transformer_spec


def _register_gather_tree_op(tf, tf_version):
    if tf_version == 1:
        from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
    elif tf_version == 2:
        # TensorFlow Addons lazily loads custom ops. So we call the op with invalid inputs
        # just to trigger the registration.
        # See also: https://github.com/tensorflow/addons/issues/1151.
        import tensorflow_addons as tfa

        try:
            tfa.seq2seq.gather_tree(0, 0, 0, 0)
        except tf.errors.InvalidArgumentError:
            pass
    else:
        raise ValueError("Unsupported TensorFlow version %d" % tf_version)


def load_model(model_path, src_vocab=None, tgt_vocab=None):
    """Loads variables and vocabularies from a TensorFlow checkpoint or SavedModel."""
    import tensorflow as tf

    def _extract_variables(structure, scope=""):
        from tensorflow.python.training.tracking import tracking

        variables = {}
        if isinstance(structure, tf.Variable):
            variables[scope] = structure
        elif isinstance(structure, list):
            for i, value in enumerate(structure):
                variables.update(_extract_variables(value, scope="%s/%d" % (scope, i)))
        elif isinstance(structure, tracking.AutoTrackable):
            for key, value in structure.__dict__.items():
                if key.startswith("_") or key == "keras_api":
                    continue
                variables.update(
                    _extract_variables(
                        value, scope="%s/%s" % (scope, key) if scope else key
                    )
                )
        return variables

    def _get_asset_path(inputter):
        lookup_table = getattr(inputter, "tokens_to_ids", None)
        if lookup_table is None:
            return None
        # TODO: retrieve the asset path without using private attributes.
        asset = getattr(lookup_table._initializer, "_filename", None)
        if asset is None:
            return None
        return asset.asset_path.numpy()

    model_version = 1
    tf_version = int(tf.version.VERSION[0])

    if tf.saved_model.contains_saved_model(model_path):
        # Force beam search kernel loading.
        _register_gather_tree_op(tf, tf_version)
        if tf_version == 2:
            model_version = 2
            imported = tf.saved_model.load(model_path)
            if src_vocab is None:
                src_vocab = _get_asset_path(
                    imported.examples_inputter.features_inputter
                )
            if tgt_vocab is None:
                tgt_vocab = _get_asset_path(imported.examples_inputter.labels_inputter)
            if src_vocab is None or tgt_vocab is None:
                raise ValueError(
                    "src_vocab and tgt_vocab are required as the SavedModel "
                    "does not include vocabulary assets"
                )
            variables = {
                "model/%s" % scope: variable.numpy()
                for scope, variable in _extract_variables(imported).items()
            }
        elif tf_version == 1:
            config = tf.compat.v1.ConfigProto(device_count={"GPU": 0})
            with tf.compat.v1.Graph().as_default():
                with tf.compat.v1.Session(config=config) as sess:
                    tf.compat.v1.saved_model.loader.load(sess, ["serve"], model_path)
                    variables = sess.run(
                        {
                            variable.op.name: variable
                            for variable in tf.compat.v1.global_variables()
                        }
                    )
                    assets = sess.run(
                        tf.compat.v1.get_collection(tf.GraphKeys.ASSET_FILEPATHS)
                    )
            src_vocab = os.path.join(
                model_path.encode("utf-8"), b"assets", os.path.basename(assets[0])
            )
            tgt_vocab = os.path.join(
                model_path.encode("utf-8"), b"assets", os.path.basename(assets[1])
            )
    else:
        if src_vocab is None or tgt_vocab is None:
            raise ValueError(
                "vocabularies must be passed as argument when converting checkpoint"
            )
        if os.path.isdir(model_path):
            checkpoint = tf.train.latest_checkpoint(model_path)
        else:
            checkpoint = model_path
        reader = tf.train.load_checkpoint(checkpoint)
        variables = {
            name: reader.get_tensor(name)
            for name in reader.get_variable_to_shape_map().keys()
        }
        if os.path.basename(checkpoint).startswith("ckpt"):
            model_version = 2
            variables = {
                name.replace("/.ATTRIBUTES/VARIABLE_VALUE", ""): value
                for name, value in variables.items()
            }
    return model_version, variables, src_vocab, tgt_vocab


class OpenNMTTFConverter(Converter):
    """Converts models generated by OpenNMT-tf."""

    def __init__(self, model_path=None, src_vocab=None, tgt_vocab=None, variables=None):
        if (model_path is None) == (variables is None):
            raise ValueError("Exactly one of model_path and variables should be set")
        if variables is not None:
            if not isinstance(variables, dict):
                raise ValueError(
                    "variables should be a dict mapping variable name to value"
                )
            if src_vocab is None or tgt_vocab is None:
                raise ValueError(
                    "src_vocab and tgt_vocab are required when directly "
                    "passing variables."
                )
        self._model_path = model_path
        self._src_vocab = src_vocab
        self._tgt_vocab = tgt_vocab
        self._variables = variables

    def _load(self, model_spec):
        if self._model_path is not None:
            version, variables, src_vocab, tgt_vocab = load_model(
                self._model_path, src_vocab=self._src_vocab, tgt_vocab=self._tgt_vocab
            )
        else:
            version = 2  # Assume we are passing V2 variables.
            variables = self._variables
            src_vocab = self._src_vocab
            tgt_vocab = self._tgt_vocab
        if isinstance(model_spec, transformer_spec.TransformerSpec):
            if version == 2:
                set_transformer_spec_v2(model_spec, variables)
            else:
                set_transformer_spec(model_spec, variables)
        else:
            raise NotImplementedError()
        return src_vocab, tgt_vocab

    def _save_vocabulary(self, vocab, destination):
        shutil.copy(vocab, destination)

    def _vocabulary_size(self, vocab):
        with open(vocab, "rb") as vocab_file:
            num_tokens = 0
            for _ in vocab_file:
                num_tokens += 1
            return num_tokens + 1  # Add OOV token.


def set_transformer_spec_v2(spec, variables):
    set_embeddings(
        spec.encoder.embeddings,
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
    set_transformer_encoder_v2(
        spec.encoder, variables, "model/encoder", relative=spec.with_relative_position
    )
    set_transformer_decoder_v2(
        spec.decoder,
        variables,
        "model/decoder",
        target_embedding_name,
        relative=spec.with_relative_position,
    )


def set_transformer_encoder_v2(spec, variables, scope, relative=False):
    set_layer_norm(spec.layer_norm, variables, "%s/layer_norm" % scope)
    for i, layer in enumerate(spec.layer):
        set_transformer_encoder_layer_v2(
            layer, variables, "%s/layers/%d" % (scope, i), relative=relative
        )


def set_transformer_decoder_v2(
    spec, variables, scope, target_embedding_name, relative=False
):
    try:
        set_linear(spec.projection, variables, "%s/output_layer" % scope)
    except KeyError:
        set_linear(
            spec.projection,
            variables,
            "%s/output_layer" % scope,
            weight_name=target_embedding_name,
            transpose=False,
        )
    set_layer_norm(spec.layer_norm, variables, "%s/layer_norm" % scope)
    for i, layer in enumerate(spec.layer):
        set_transformer_decoder_layer_v2(
            layer, variables, "%s/layers/%d" % (scope, i), relative=relative
        )


def set_transformer_encoder_layer_v2(spec, variables, scope, relative=False):
    set_ffn_v2(spec.ffn, variables, "%s/ffn" % scope)
    set_multi_head_attention_v2(
        spec.self_attention,
        variables,
        "%s/self_attention" % scope,
        self_attention=True,
        relative=relative,
    )


def set_transformer_decoder_layer_v2(spec, variables, scope, relative=False):
    set_ffn_v2(spec.ffn, variables, "%s/ffn" % scope)
    set_multi_head_attention_v2(
        spec.self_attention,
        variables,
        "%s/self_attention" % scope,
        self_attention=True,
        relative=relative,
    )
    set_multi_head_attention_v2(
        spec.attention, variables, "%s/attention/0" % scope, relative=relative
    )


def set_ffn_v2(spec, variables, scope):
    set_layer_norm(spec.layer_norm, variables, "%s/input_layer_norm" % scope)
    set_linear(spec.linear_0, variables, "%s/layer/inner" % scope)
    set_linear(spec.linear_1, variables, "%s/layer/outer" % scope)


def set_multi_head_attention_v2(
    spec, variables, scope, self_attention=False, relative=False
):
    set_layer_norm(spec.layer_norm, variables, "%s/input_layer_norm" % scope)
    if self_attention:
        split_layers = [common_spec.LinearSpec() for _ in range(3)]
        set_linear(split_layers[0], variables, "%s/layer/linear_queries" % scope)
        set_linear(split_layers[1], variables, "%s/layer/linear_keys" % scope)
        set_linear(split_layers[2], variables, "%s/layer/linear_values" % scope)
        utils.fuse_linear(spec.linear[0], split_layers)
        if relative:
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
        set_embeddings(spec.embeddings, variables, "transformer/encoder")
    except KeyError:
        # Try shared embeddings scope instead.
        set_embeddings(spec.embeddings, variables, "transformer/shared_embeddings")
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
    spec.multiply_by_sqrt_depth = True
    return variable_name
