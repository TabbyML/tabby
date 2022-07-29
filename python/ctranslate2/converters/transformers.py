import abc
import argparse

import numpy as np

from ctranslate2.converters import utils
from ctranslate2.converters.converter import Converter
from ctranslate2.specs import common_spec, model_spec, transformer_spec

_SUPPORTED_ACTIVATIONS = {
    "gelu": common_spec.Activation.GELU,
    "gelu_fast": common_spec.Activation.GELU,
    "gelu_new": common_spec.Activation.GELU,
    "gelu_python": common_spec.Activation.GELU,
    "quick_gelu": common_spec.Activation.GELU,
    "relu": common_spec.Activation.RELU,
    "silu": common_spec.Activation.SWISH,
    "swish": common_spec.Activation.SWISH,
}

_MODEL_LOADERS = {}


def register_loader(config_name):
    """Registers a model loader for this configuration name."""

    def decorator(cls):
        _MODEL_LOADERS[config_name] = cls()
        return cls

    return decorator


class TransformersConverter(Converter):
    """Converts models from Hugging Face Transformers."""

    def __init__(self, model_name_or_path: str):
        """Initializes the converter.

        Arguments:
          model_name_or_path: Name of the pretrained model to download, or path to the
            directory containing the pretrained model.
        """
        self._model_name_or_path = model_name_or_path

    def _load(self):
        import torch
        import transformers

        with torch.no_grad():
            config = transformers.AutoConfig.from_pretrained(self._model_name_or_path)
            config_name = config.__class__.__name__
            loader = _MODEL_LOADERS.get(config_name)

            if loader is None:
                raise ValueError(
                    "No conversion is registered for the model configuration %s "
                    "(supported configurations are: %s)"
                    % (config_name, ", ".join(_MODEL_LOADERS.keys()))
                )

            return loader(self._model_name_or_path)


class ModelLoader(abc.ABC):
    """Base class for loading Transformers models into a CTranslate2 model specification."""

    @property
    def architecture_name(self):
        return None

    @abc.abstractmethod
    def get_model_spec(self, model):
        raise NotImplementedError()

    def __call__(self, model_name_or_path):
        import transformers

        model_class = getattr(transformers, self.architecture_name)
        model = model_class.from_pretrained(model_name_or_path)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=False
        )

        spec = self.get_model_spec(model)

        tokens = self.get_vocabulary(model, tokenizer)
        if model.config.vocab_size < len(tokens):
            tokens = tokens[: model.config.vocab_size]
        if isinstance(spec, model_spec.SequenceToSequenceModelSpec):
            spec.register_source_vocabulary(tokens)
            spec.register_target_vocabulary(tokens)
        else:
            spec.register_vocabulary(tokens)

        if tokenizer.bos_token is not None:
            spec.bos_token = tokenizer.bos_token
        if tokenizer.eos_token is not None:
            spec.eos_token = tokenizer.eos_token
        if tokenizer.unk_token is not None:
            spec.unk_token = tokenizer.unk_token

        return spec

    def get_vocabulary(self, model, tokenizer):
        return [
            token
            for token, _ in sorted(
                tokenizer.get_vocab().items(), key=lambda item: item[1]
            )
        ]

    def set_layer_norm(self, spec, module):
        spec.gamma = module.weight.numpy()
        spec.beta = module.bias.numpy()

    def set_linear(self, spec, module):
        import transformers

        spec.weight = module.weight.numpy()
        if isinstance(module, transformers.Conv1D):
            spec.weight = spec.weight.transpose()
        if module.bias is not None:
            spec.bias = module.bias.numpy()

    def set_embeddings(self, spec, module):
        spec.weight = module.weight.numpy()

    def set_position_encodings(self, spec, module):
        spec.encodings = module.weight.numpy()
        offset = getattr(module, "offset", 0)
        if offset > 0:
            spec.encodings = spec.encodings[offset:]


@register_loader("BartConfig")
class BartLoader(ModelLoader):
    @property
    def architecture_name(self):
        return "BartForConditionalGeneration"

    def get_model_spec(self, model):
        spec = transformer_spec.TransformerSpec(
            model.config.encoder_layers,
            model.config.encoder_attention_heads,
            pre_norm=model.config.normalize_before,
            activation=_SUPPORTED_ACTIVATIONS[model.config.activation_function],
            layernorm_embedding=getattr(model.config, "normalize_embedding", True),
        )
        spec.with_target_bos = False

        self.set_encoder(spec.encoder, model.model.encoder)
        self.set_decoder(spec.decoder, model.model.decoder)
        self.set_linear(spec.decoder.projection, model.lm_head)

        final_logits_bias = getattr(model, "final_logits_bias", None)
        if final_logits_bias is not None and final_logits_bias.nonzero().numel() != 0:
            spec.decoder.projection.bias = final_logits_bias.squeeze().numpy()

        return spec

    def set_encoder(self, spec, encoder):
        self.set_common_layers(spec, encoder)

        for layer_spec, layer in zip(spec.layer, encoder.layers):
            self.set_attention(
                layer_spec.self_attention,
                layer.self_attn,
                self_attention=True,
            )
            self.set_layer_norm(
                layer_spec.self_attention.layer_norm,
                layer.self_attn_layer_norm,
            )

            self.set_linear(layer_spec.ffn.linear_0, layer.fc1)
            self.set_linear(layer_spec.ffn.linear_1, layer.fc2)
            self.set_layer_norm(layer_spec.ffn.layer_norm, layer.final_layer_norm)

    def set_decoder(self, spec, decoder):
        self.set_common_layers(spec, decoder)

        for layer_spec, layer in zip(spec.layer, decoder.layers):
            self.set_attention(
                layer_spec.self_attention,
                layer.self_attn,
                self_attention=True,
            )
            self.set_layer_norm(
                layer_spec.self_attention.layer_norm,
                layer.self_attn_layer_norm,
            )

            if hasattr(layer, "encoder_attn"):
                self.set_attention(
                    layer_spec.attention,
                    layer.encoder_attn,
                    self_attention=False,
                )
                self.set_layer_norm(
                    layer_spec.attention.layer_norm,
                    layer.encoder_attn_layer_norm,
                )

            self.set_linear(layer_spec.ffn.linear_0, layer.fc1)
            self.set_linear(layer_spec.ffn.linear_1, layer.fc2)
            self.set_layer_norm(layer_spec.ffn.layer_norm, layer.final_layer_norm)

    def set_attention(self, spec, attention, self_attention=False):
        split_layers = [common_spec.LinearSpec() for _ in range(3)]
        self.set_linear(split_layers[0], attention.q_proj)
        self.set_linear(split_layers[1], attention.k_proj)
        self.set_linear(split_layers[2], attention.v_proj)

        if self_attention:
            utils.fuse_linear(spec.linear[0], split_layers)
        else:
            utils.fuse_linear(spec.linear[0], split_layers[:1])
            utils.fuse_linear(spec.linear[1], split_layers[1:])

        self.set_linear(spec.linear[-1], attention.out_proj)

    def set_common_layers(self, spec, module):
        spec.scale_embeddings = module.embed_scale
        self.set_position_encodings(spec.position_encodings, module.embed_positions)
        self.set_embeddings(
            spec.embeddings[0]
            if isinstance(spec.embeddings, list)
            else spec.embeddings,
            module.embed_tokens,
        )

        if hasattr(module, "layer_norm"):
            self.set_layer_norm(spec.layer_norm, module.layer_norm)
        if hasattr(module, "layernorm_embedding"):
            self.set_layer_norm(spec.layernorm_embedding, module.layernorm_embedding)


@register_loader("MarianConfig")
class MarianMTLoader(BartLoader):
    @property
    def architecture_name(self):
        return "MarianMTModel"

    def get_model_spec(self, model):
        model.config.normalize_before = False
        model.config.normalize_embedding = False
        spec = super().get_model_spec(model)
        self._remove_pad_weights(spec)
        return spec

    def set_decoder(self, spec, decoder):
        spec.start_from_zero_embedding = True
        super().set_decoder(spec, decoder)

    def get_vocabulary(self, model, tokenizer):
        # The <pad> token is added by Transformers to start the decoder from a zero embedding,
        # but we already have a dedicated option "start_from_zero_embedding". We remove this token
        # to match the original Marian vocabulary and prevent this token from being generated.
        tokens = super().get_vocabulary(model, tokenizer)
        if tokens[-1] == "<pad>":
            tokens.pop()
        return tokens

    def _remove_pad_weights(self, spec):
        vocab_specs = [
            spec.encoder.embeddings[0],
            spec.decoder.embeddings,
            spec.decoder.projection,
        ]

        # Weights may be shared so we check against the expected size to prevent
        # updating the same weight multiple times.
        new_vocab_size = vocab_specs[0].weight.shape[0] - 1

        for vocab_spec in vocab_specs:
            if vocab_spec.weight.shape[0] == new_vocab_size + 1:
                vocab_spec.weight = vocab_spec.weight[:-1]
            if (
                isinstance(vocab_spec, common_spec.LinearSpec)
                and isinstance(vocab_spec.bias, np.ndarray)
                and vocab_spec.bias.shape[0] == new_vocab_size + 1
            ):
                vocab_spec.bias = vocab_spec.bias[:-1]


@register_loader("M2M100Config")
class M2M100Loader(BartLoader):
    @property
    def architecture_name(self):
        return "M2M100ForConditionalGeneration"

    def get_model_spec(self, model):
        model.config.normalize_before = True
        model.config.normalize_embedding = False
        return super().get_model_spec(model)

    def set_position_encodings(self, spec, module):
        spec.encodings = module.weights.numpy()[module.offset :]

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)

        for token in tokenizer.additional_special_tokens:
            if token not in tokens:
                tokens.append(token)

        num_madeup_words = getattr(
            tokenizer, "num_madeup_words", model.config.vocab_size - len(tokens)
        )
        if num_madeup_words > 0:
            tokens += ["madeupword%d" % i for i in range(num_madeup_words)]

        return tokens


@register_loader("MBartConfig")
class MBartLoader(BartLoader):
    @property
    def architecture_name(self):
        return "MBartForConditionalGeneration"

    def get_model_spec(self, model):
        spec = super().get_model_spec(model)

        # MBart-25 passes the language code as the decoder start token.
        if model.config.tokenizer_class in ("MBartTokenizer", None):
            spec.user_decoder_start_tokens = True

        return spec


@register_loader("PegasusConfig")
class PegasusLoader(BartLoader):
    @property
    def architecture_name(self):
        return "PegasusForConditionalGeneration"

    def get_model_spec(self, model):
        spec = super().get_model_spec(model)
        spec.with_target_bos = True
        return spec

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)
        tokenizer.bos_token = tokens[model.config.pad_token_id]
        return tokens


@register_loader("OPTConfig")
class OPTLoader(BartLoader):
    @property
    def architecture_name(self):
        return "OPTForCausalLM"

    def get_model_spec(self, model):
        spec = transformer_spec.TransformerDecoderModelSpec(
            model.config.num_hidden_layers,
            model.config.num_attention_heads,
            pre_norm=model.config.do_layer_norm_before,
            activation=_SUPPORTED_ACTIVATIONS[model.config.activation_function],
            project_in_out=model.config.word_embed_proj_dim != model.config.hidden_size,
        )

        self.set_decoder(spec.decoder, model.model.decoder)
        self.set_linear(spec.decoder.projection, model.lm_head)
        return spec

    def set_decoder(self, spec, decoder):
        super().set_decoder(spec, decoder)

        if decoder.project_in is not None:
            self.set_linear(spec.project_in, decoder.project_in)
        if decoder.project_out is not None:
            self.set_linear(spec.project_out, decoder.project_out)
        if decoder.final_layer_norm is not None:
            self.set_layer_norm(spec.layer_norm, decoder.final_layer_norm)

    def set_common_layers(self, spec, module):
        spec.scale_embeddings = False
        self.set_position_encodings(spec.position_encodings, module.embed_positions)
        self.set_embeddings(spec.embeddings, module.embed_tokens)

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)

        i = 0
        while len(tokens) % 8 != 0:
            symbol = "madeupword{:04d}".format(i)
            if symbol not in tokens:
                tokens.append(symbol)
            i += 1

        return tokens


@register_loader("GPT2Config")
class GPT2Loader(ModelLoader):
    @property
    def architecture_name(self):
        return "GPT2LMHeadModel"

    def get_model_spec(self, model):
        spec = transformer_spec.TransformerDecoderModelSpec(
            model.config.n_layer,
            model.config.n_head,
            pre_norm=True,
            activation=_SUPPORTED_ACTIVATIONS[model.config.activation_function],
        )

        self.set_decoder(spec.decoder, model.transformer)
        self.set_linear(spec.decoder.projection, model.lm_head)
        return spec

    def set_decoder(self, spec, module):
        spec.scale_embeddings = False
        self.set_embeddings(spec.embeddings, module.wte)
        self.set_position_encodings(spec.position_encodings, module.wpe)
        self.set_layer_norm(spec.layer_norm, module.ln_f)

        for layer_spec, layer in zip(spec.layer, module.h):
            self.set_layer_norm(layer_spec.self_attention.layer_norm, layer.ln_1)
            self.set_linear(layer_spec.self_attention.linear[0], layer.attn.c_attn)
            self.set_linear(layer_spec.self_attention.linear[1], layer.attn.c_proj)
            self.set_layer_norm(layer_spec.ffn.layer_norm, layer.ln_2)
            self.set_linear(layer_spec.ffn.linear_0, layer.mlp.c_fc)
            self.set_linear(layer_spec.ffn.linear_1, layer.mlp.c_proj)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Name of the pretrained model to download, "
            "or path to a directory containing the pretrained model."
        ),
    )

    Converter.declare_arguments(parser)
    args = parser.parse_args()
    converter = TransformersConverter(args.model)
    converter.convert_from_args(args)


if __name__ == "__main__":
    main()
