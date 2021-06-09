from ctranslate2.converters import utils
from ctranslate2.converters.converter import Converter
from ctranslate2.specs import common_spec
from ctranslate2.specs import transformer_spec


_SUPPORTED_ARCHS = {
    "transformer",
    "transformer_iwslt_de_en",
    "transformer_tiny",
    "transformer_vaswani_wmt_en_de_big",
    "transformer_vaswani_wmt_en_fr_big",
    "transformer_wmt_en_de",
    "transformer_wmt_en_de_big",
    "transformer_wmt_en_de_big_t2t",
}


def _check_args(args):
    reasons = []
    if args.arch not in _SUPPORTED_ARCHS:
        reasons.append(
            "Option --arch %s is not supported (supported architectures are: %s)"
            % (args.arch, ", ".join(_SUPPORTED_ARCHS))
        )
    if args.encoder_normalize_before != args.decoder_normalize_before:
        reasons.append(
            "Options --encoder-normalize-before and --decoder-normalize-before "
            "must have the same value"
        )
    if args.encoder_attention_heads != args.decoder_attention_heads:
        reasons.append(
            "Options --encoder-attention-heads and --decoder-attention-heads must "
            "have the same value"
        )
    if getattr(args, "activation_fn", "relu") != "relu":
        reasons.append(
            "Option --activation-fn %s is not supported (supported activations are: relu)"
            % args.activation_fn
        )
    if getattr(args, "no_token_positional_embeddings", False):
        reasons.append("Option --no-token-positional-embeddings is not supported")
    if getattr(args, "layernorm_embedding", False):
        reasons.append("Option --layernorm-embedding is not supported")

    if reasons:
        utils.raise_unsupported(reasons)


def _get_model_spec(args):
    _check_args(args)
    return transformer_spec.TransformerSpec(
        (args.encoder_layers, args.decoder_layers),
        args.encoder_attention_heads,
        pre_norm=args.encoder_normalize_before,
    )


def _get_vocab(dictionary):
    return ["<blank>" if token == "<pad>" else token for token in dictionary.symbols]


class FairseqConverter(Converter):
    """Converts models trained with Fairseq."""

    def __init__(self, model_path, data_dir):
        self._model_path = model_path
        self._data_dir = data_dir

    def _load(self):
        import torch
        import fairseq
        from fairseq import checkpoint_utils

        with torch.no_grad():
            checkpoint = checkpoint_utils.load_checkpoint_to_cpu(self._model_path)
            args = checkpoint["args"]
            args.data = self._data_dir

            model_spec = _get_model_spec(args)
            model_spec.with_source_eos = True
            model_spec.with_target_bos = False

            task = fairseq.tasks.setup_task(args)
            model = fairseq.models.build_model(args, task)
            model.load_state_dict(checkpoint["model"])

            set_transformer_spec(model_spec, model)
            model_spec.register_vocabulary("source", _get_vocab(task.source_dictionary))
            model_spec.register_vocabulary("target", _get_vocab(task.target_dictionary))
            return model_spec


def set_transformer_spec(spec, module):
    set_transformer_encoder(spec.encoder, module.encoder)
    set_transformer_decoder(spec.decoder, module.decoder)


def set_transformer_encoder(spec, module):
    set_input_layers(spec, module)
    for layer_spec, layer in zip(spec.layer, module.layers):
        set_transformer_encoder_layer(layer_spec, layer)
    if module.layer_norm is not None:
        set_layer_norm(spec.layer_norm, module.layer_norm)


def set_transformer_decoder(spec, module):
    set_input_layers(spec, module)
    set_linear(spec.projection, module.output_projection)
    for layer_spec, layer in zip(spec.layer, module.layers):
        set_transformer_decoder_layer(layer_spec, layer)
    if module.layer_norm is not None:
        set_layer_norm(spec.layer_norm, module.layer_norm)


def set_input_layers(spec, module):
    set_position_encodings(spec.position_encodings, module.embed_positions)
    set_embeddings(
        spec.embeddings,
        module.embed_tokens,
        multiply_by_sqrt_depth=module.embed_scale != 1.0,
    )


def set_transformer_encoder_layer(spec, module):
    set_ffn(spec.ffn, module)
    set_multi_head_attention(spec.self_attention, module.self_attn, self_attention=True)
    set_layer_norm(spec.self_attention.layer_norm, module.self_attn_layer_norm)


def set_transformer_decoder_layer(spec, module):
    set_ffn(spec.ffn, module)
    set_multi_head_attention(spec.self_attention, module.self_attn, self_attention=True)
    set_layer_norm(spec.self_attention.layer_norm, module.self_attn_layer_norm)
    set_multi_head_attention(spec.attention, module.encoder_attn)
    set_layer_norm(spec.attention.layer_norm, module.encoder_attn_layer_norm)


def set_ffn(spec, module):
    set_layer_norm(spec.layer_norm, module.final_layer_norm)
    set_linear(spec.linear_0, module.fc1)
    set_linear(spec.linear_1, module.fc2)


def set_multi_head_attention(spec, module, self_attention=False):
    if self_attention:
        split_layers = [common_spec.LinearSpec() for _ in range(3)]
        set_linear(split_layers[0], module.q_proj)
        set_linear(split_layers[1], module.k_proj)
        set_linear(split_layers[2], module.v_proj)
        utils.fuse_linear(spec.linear[0], split_layers)
    else:
        set_linear(spec.linear[0], module.q_proj)
        split_layers = [common_spec.LinearSpec() for _ in range(2)]
        set_linear(split_layers[0], module.k_proj)
        set_linear(split_layers[1], module.v_proj)
        utils.fuse_linear(spec.linear[1], split_layers)
    set_linear(spec.linear[-1], module.out_proj)


def set_layer_norm(spec, module):
    spec.gamma = module.weight.numpy()
    spec.beta = module.bias.numpy()


def set_linear(spec, module):
    spec.weight = module.weight.numpy()
    if module.bias is not None:
        spec.bias = module.bias.numpy()


def set_embeddings(spec, module, multiply_by_sqrt_depth=True):
    spec.weight = module.weight.numpy()
    spec.multiply_by_sqrt_depth = multiply_by_sqrt_depth


def set_position_encodings(spec, module):
    spec.encodings = module.weights.numpy()[module.padding_idx + 1 :]
