import argparse
import re

import numpy as np
import yaml

from ctranslate2.converters import utils
from ctranslate2.converters.converter import Converter
from ctranslate2.specs import common_spec, model_spec, transformer_spec

_SUPPORTED_ACTIVATIONS = {
    "gelu": common_spec.Activation.GELU,
    "relu": common_spec.Activation.RELU,
    "swish": common_spec.Activation.SWISH,
}

_SUPPORTED_POSTPROCESS_EMB = {"", "d", "n", "nd"}


class MarianConverter(Converter):
    """Converts models trained with Marian."""

    def __init__(self, model_path, vocab_paths):
        self._model_path = model_path
        self._vocab_paths = vocab_paths

    def _load(self):
        model = np.load(self._model_path)
        config = _get_model_config(model)
        vocabs = list(map(_load_vocab, self._vocab_paths))

        activation = config["transformer-ffn-activation"]
        pre_norm = "n" in config["transformer-preprocess"]
        postprocess_emb = config["transformer-postprocess-emb"]

        check = utils.ConfigurationChecker()
        check(config["type"] == "transformer", "Option --type must be 'transformer'")
        check(
            config["transformer-decoder-autoreg"] == "self-attention",
            "Option --transformer-decoder-autoreg must be 'self-attention'",
        )
        check(
            not config["transformer-no-projection"],
            "Option --transformer-no-projection is not supported",
        )
        check(
            activation in _SUPPORTED_ACTIVATIONS,
            "Option --transformer-ffn-activation %s is not supported "
            "(supported activations are: %s)"
            % (activation, ", ".join(_SUPPORTED_ACTIVATIONS.keys())),
        )
        check(
            postprocess_emb in _SUPPORTED_POSTPROCESS_EMB,
            "Option --transformer-postprocess-emb %s is not supported (supported values are: %s)"
            % (postprocess_emb, ", ".join(_SUPPORTED_POSTPROCESS_EMB)),
        )

        if pre_norm:
            check(
                config["transformer-preprocess"] == "n"
                and config["transformer-postprocess"] == "da"
                and config.get("transformer-postprocess-top", "") == "n",
                "Unsupported pre-norm Transformer architecture, expected the following "
                "combination of options: "
                "--transformer-preprocess n "
                "--transformer-postprocess da "
                "--transformer-postprocess-top n",
            )
        else:
            check(
                config["transformer-preprocess"] == ""
                and config["transformer-postprocess"] == "dan"
                and config.get("transformer-postprocess-top", "") == "",
                "Unsupported post-norm Transformer architecture, excepted the following "
                "combination of options: "
                "--transformer-preprocess '' "
                "--transformer-postprocess dan "
                "--transformer-postprocess-top ''",
            )

        check.validate()

        alignment_layer = config["transformer-guided-alignment-layer"]
        alignment_layer = -1 if alignment_layer == "last" else int(alignment_layer) - 1
        layernorm_embedding = "n" in postprocess_emb

        model_spec = transformer_spec.TransformerSpec(
            (config["enc-depth"], config["dec-depth"]),
            config["transformer-heads"],
            pre_norm=pre_norm,
            activation=_SUPPORTED_ACTIVATIONS[activation],
            alignment_layer=alignment_layer,
            alignment_heads=1,
            layernorm_embedding=layernorm_embedding,
        )
        set_transformer_spec(model_spec, model)
        model_spec.register_source_vocabulary(vocabs[0])
        model_spec.register_target_vocabulary(vocabs[-1])
        model_spec.with_source_eos = True
        return model_spec


def _get_model_config(model):
    config = model["special:model.yml"]
    config = config[:-1].tobytes()
    config = yaml.safe_load(config)
    return config


def _load_vocab(path):
    # pyyaml skips some entries so we manually parse the vocabulary file.
    with open(path, encoding="utf-8") as vocab:
        tokens = []
        for i, line in enumerate(vocab):
            line = line.rstrip("\n\r")
            token, idx = line.rsplit(":", 1)
            try:
                int(idx.strip())
            except ValueError as e:
                raise ValueError(
                    "Unexpected format at line %d: '%s'" % (i + 1, line)
                ) from e
            if token.startswith('"') and token.endswith('"'):
                # Unescape characters and remove quotes.
                token = re.sub(r"\\(.)", r"\1", token)
                token = token[1:-1]
            tokens.append(token)
        return tokens


def set_transformer_spec(spec, weights):
    set_transformer_encoder(spec.encoder, weights, "encoder")
    set_transformer_decoder(spec.decoder, weights, "decoder")


def set_transformer_encoder(spec, weights, scope):
    set_common_layers(spec, weights, scope)
    for i, layer_spec in enumerate(spec.layer):
        set_transformer_encoder_layer(layer_spec, weights, "%s_l%d" % (scope, i + 1))


def set_transformer_decoder(spec, weights, scope):
    spec.start_from_zero_embedding = True
    set_common_layers(spec, weights, scope)
    for i, layer_spec in enumerate(spec.layer):
        set_transformer_decoder_layer(layer_spec, weights, "%s_l%d" % (scope, i + 1))

    set_linear(
        spec.projection,
        weights,
        "%s_ff_logit_out" % scope,
        reuse_weight=spec.embeddings.weight,
    )


def set_common_layers(spec, weights, scope):
    embeddings_specs = spec.embeddings
    if not isinstance(embeddings_specs, list):
        embeddings_specs = [embeddings_specs]

    set_embeddings(embeddings_specs[0], weights, scope)
    set_position_encodings(
        spec.position_encodings, weights, dim=embeddings_specs[0].weight.shape[1]
    )
    if spec.layernorm_embedding != model_spec.OPTIONAL:
        set_layer_norm(spec.layernorm_embedding, weights, "%s_emb" % scope)
    if spec.layer_norm != model_spec.OPTIONAL:
        set_layer_norm(spec.layer_norm, weights, "%s_top" % scope)


def set_transformer_encoder_layer(spec, weights, scope):
    set_ffn(spec.ffn, weights, "%s_ffn" % scope)
    set_multi_head_attention(
        spec.self_attention, weights, "%s_self" % scope, self_attention=True
    )


def set_transformer_decoder_layer(spec, weights, scope):
    set_ffn(spec.ffn, weights, "%s_ffn" % scope)
    set_multi_head_attention(
        spec.self_attention, weights, "%s_self" % scope, self_attention=True
    )
    set_multi_head_attention(spec.attention, weights, "%s_context" % scope)


def set_multi_head_attention(spec, weights, scope, self_attention=False):
    split_layers = [common_spec.LinearSpec() for _ in range(3)]
    set_linear(split_layers[0], weights, scope, "q")
    set_linear(split_layers[1], weights, scope, "k")
    set_linear(split_layers[2], weights, scope, "v")

    if self_attention:
        utils.fuse_linear(spec.linear[0], split_layers)
    else:
        spec.linear[0].weight = split_layers[0].weight
        spec.linear[0].bias = split_layers[0].bias
        utils.fuse_linear(spec.linear[1], split_layers[1:])

    set_linear(spec.linear[-1], weights, scope, "o")
    set_layer_norm(spec.layer_norm, weights, "%s_Wo" % scope)


def set_ffn(spec, weights, scope):
    set_layer_norm(spec.layer_norm, weights, "%s_ffn" % scope)
    set_linear(spec.linear_0, weights, scope, "1")
    set_linear(spec.linear_1, weights, scope, "2")


def set_layer_norm(spec, weights, scope):
    spec.gamma = weights["%s_ln_scale" % scope].squeeze()
    spec.beta = weights["%s_ln_bias" % scope].squeeze()


def set_linear(spec, weights, scope, suffix="", reuse_weight=None):
    weight = weights.get("%s_W%s" % (scope, suffix))
    spec.weight = reuse_weight if weight is None else weight.transpose()

    bias = weights.get("%s_b%s" % (scope, suffix))
    if bias is not None:
        spec.bias = bias.squeeze()


def set_embeddings(spec, weights, scope):
    spec.weight = weights.get("%s_Wemb" % scope)
    if spec.weight is None:
        spec.weight = weights.get("Wemb")


def set_position_encodings(spec, weights, dim=None):
    spec.encodings = weights.get("Wpos", _make_sinusoidal_position_encodings(dim))


def _make_sinusoidal_position_encodings(dim, num_positions=2048):
    # Copied from https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/models/marian/modeling_marian.py  # noqa: E501
    position_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
            for pos in range(num_positions)
        ]
    )
    table = np.zeros_like(position_enc)
    table[:, : dim // 2] = np.sin(position_enc[:, 0::2])
    table[:, dim // 2 :] = np.cos(position_enc[:, 1::2])
    return table


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_path", required=True, help="Path to the model .npz file."
    )
    parser.add_argument(
        "--vocab_paths",
        required=True,
        nargs="+",
        help="List of paths to the YAML vocabularies.",
    )
    Converter.declare_arguments(parser)
    args = parser.parse_args()
    converter = MarianConverter(args.model_path, args.vocab_paths)
    converter.convert_from_args(args)


if __name__ == "__main__":
    main()
