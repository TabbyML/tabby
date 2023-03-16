import abc
import argparse
import itertools
import os

from typing import List, Optional

import numpy as np

from ctranslate2.converters import utils
from ctranslate2.converters.converter import Converter
from ctranslate2.specs import common_spec, model_spec, transformer_spec, whisper_spec

_SUPPORTED_ACTIVATIONS = {
    "gelu": common_spec.Activation.GELU,
    "gelu_fast": common_spec.Activation.GELUTanh,
    "gelu_new": common_spec.Activation.GELUTanh,
    "gelu_python": common_spec.Activation.GELU,
    "gelu_pytorch_tanh": common_spec.Activation.GELUTanh,
    "quick_gelu": common_spec.Activation.GELUSigmoid,
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

    def __init__(
        self,
        model_name_or_path: str,
        activation_scales: Optional[str] = None,
        copy_files: Optional[List[str]] = None,
        load_as_float16: bool = False,
    ):
        """Initializes the converter.

        Arguments:
          model_name_or_path: Name of the pretrained model to download, or path to the
            directory containing the pretrained model.
          activation_scales: Path to the pre-computed activation scales. Models may
            use them to rescale some weights to smooth the intermediate activations
            and improve the quantization accuracy. See
            https://github.com/mit-han-lab/smoothquant.
          copy_files: List of filenames to copy from the Hugging Face model to the
            converted model directory.
          load_as_float16: Load the model weights as float16. More precisely, the model
            will be loaded with ``from_pretrained(..., torch_dtype=torch.float16)``.
        """
        self._model_name_or_path = model_name_or_path
        self._activation_scales = activation_scales
        self._copy_files = copy_files
        self._load_as_float16 = load_as_float16

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

            model_class = getattr(transformers, loader.architecture_name)
            tokenizer_class = transformers.AutoTokenizer

            torch_dtype = torch.float16 if self._load_as_float16 else None
            model = self.load_model(
                model_class,
                self._model_name_or_path,
                torch_dtype=torch_dtype,
            )
            tokenizer = self.load_tokenizer(
                tokenizer_class,
                self._model_name_or_path,
                use_fast=False,
            )

            spec = loader(model, tokenizer)

            if self._activation_scales:
                activation_scales = torch.load(
                    self._activation_scales, map_location="cpu"
                )
                loader.smooth_activation(spec, activation_scales)

            if self._copy_files:
                for filename in self._copy_files:
                    spec.register_file(self.get_model_file(filename))

            return spec

    def load_model(self, model_class, model_name_or_path, **kwargs):
        return model_class.from_pretrained(model_name_or_path, **kwargs)

    def load_tokenizer(self, tokenizer_class, model_name_or_path, **kwargs):
        return tokenizer_class.from_pretrained(model_name_or_path, **kwargs)

    def get_model_file(self, filename):
        if os.path.isdir(self._model_name_or_path):
            path = os.path.join(self._model_name_or_path, filename)
        else:
            import huggingface_hub

            try:
                path = huggingface_hub.hf_hub_download(
                    repo_id=self._model_name_or_path, filename=filename
                )
            except huggingface_hub.utils.EntryNotFoundError:
                path = None

        if path is None or not os.path.isfile(path):
            raise ValueError(
                "File %s does not exist in model %s"
                % (filename, self._model_name_or_path)
            )

        return path


class ModelLoader(abc.ABC):
    """Base class for loading Transformers models into a CTranslate2 model specification."""

    @property
    def architecture_name(self):
        return None

    @abc.abstractmethod
    def get_model_spec(self, model):
        raise NotImplementedError()

    def __call__(self, model, tokenizer):
        spec = self.get_model_spec(model)
        self.set_config(spec.config, model, tokenizer)

        tokens = self.get_vocabulary(model, tokenizer)
        self.set_vocabulary(spec, tokens)

        return spec

    def get_vocabulary(self, model, tokenizer):
        return [
            token
            for token, _ in sorted(
                tokenizer.get_vocab().items(), key=lambda item: item[1]
            )
        ]

    def set_vocabulary(self, spec, tokens):
        pass

    def set_config(self, config, model, tokenizer):
        pass

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

    def smooth_activation(self, spec, activation_scales):
        raise NotImplementedError(
            "No activation smoothing logic is defined for this model"
        )


@register_loader("BartConfig")
class BartLoader(ModelLoader):
    @property
    def architecture_name(self):
        return "BartForConditionalGeneration"

    def get_model_spec(self, model):
        spec = transformer_spec.TransformerSpec.from_config(
            (model.config.encoder_layers, model.config.decoder_layers),
            model.config.encoder_attention_heads,
            pre_norm=model.config.normalize_before,
            activation=_SUPPORTED_ACTIVATIONS[model.config.activation_function],
            layernorm_embedding=getattr(model.config, "normalize_embedding", True),
        )

        self.set_encoder(spec.encoder, model.model.encoder)
        self.set_decoder(spec.decoder, model.model.decoder)
        self.set_linear(spec.decoder.projection, model.lm_head)

        final_logits_bias = getattr(model, "final_logits_bias", None)
        if final_logits_bias is not None and final_logits_bias.nonzero().numel() != 0:
            spec.decoder.projection.bias = final_logits_bias.squeeze().numpy()

        return spec

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)
        if model.config.vocab_size < len(tokens):
            tokens = tokens[: model.config.vocab_size]
        return tokens

    def set_vocabulary(self, spec, tokens):
        spec.register_source_vocabulary(tokens)
        spec.register_target_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.bos_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token
        config.decoder_start_token = tokenizer.convert_ids_to_tokens(
            model.config.decoder_start_token_id
        )

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

    def set_config(self, config, model, tokenizer):
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token

        # The decoder start token can be any token because the decoder always starts
        # from a zero embedding.
        config.decoder_start_token = tokenizer.eos_token

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

        # Workaround for issue https://github.com/OpenNMT/CTranslate2/issues/1039.
        if tokens[-1] == tokenizer.unk_token:
            tokens.insert(tokenizer.unk_token_id, tokens.pop())

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

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.bos_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token

        # MBart-25 passes the language code as the decoder start token.
        if model.config.tokenizer_class in ("MBartTokenizer", None):
            config.decoder_start_token = None
        else:
            config.decoder_start_token = tokenizer.eos_token


@register_loader("PegasusConfig")
class PegasusLoader(BartLoader):
    @property
    def architecture_name(self):
        return "PegasusForConditionalGeneration"

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.pad_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token
        config.decoder_start_token = tokenizer.pad_token


@register_loader("OPTConfig")
class OPTLoader(BartLoader):
    @property
    def architecture_name(self):
        return "OPTForCausalLM"

    def get_model_spec(self, model):
        spec = transformer_spec.TransformerDecoderModelSpec.from_config(
            model.config.num_hidden_layers,
            model.config.num_attention_heads,
            pre_norm=model.config.do_layer_norm_before,
            activation=_SUPPORTED_ACTIVATIONS[model.config.activation_function],
            project_in_out=model.config.word_embed_proj_dim != model.config.hidden_size,
        )

        self.set_decoder(spec.decoder, model.model.decoder)
        self.set_linear(spec.decoder.projection, model.lm_head)
        return spec

    def smooth_activation(self, spec, activation_scales):
        for i, layer in enumerate(spec.decoder.layer):
            layer_scope = "model.decoder.layers.%d" % i

            utils.smooth_activation(
                layer.self_attention.layer_norm,
                layer.self_attention.linear[0],
                activation_scales["%s.self_attn.q_proj" % layer_scope].numpy(),
            )

            utils.smooth_activation(
                layer.ffn.layer_norm,
                layer.ffn.linear_0,
                activation_scales["%s.fc1" % layer_scope].numpy(),
            )

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.bos_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token

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
        spec = transformer_spec.TransformerDecoderModelSpec.from_config(
            model.config.n_layer,
            model.config.n_head,
            pre_norm=True,
            activation=_SUPPORTED_ACTIVATIONS[model.config.activation_function],
        )

        self.set_decoder(spec.decoder, model.transformer)
        self.set_linear(spec.decoder.projection, model.lm_head)
        return spec

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.bos_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token

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


@register_loader("WhisperConfig")
class WhisperLoader(BartLoader):
    @property
    def architecture_name(self):
        return "WhisperForConditionalGeneration"

    def get_model_spec(self, model):
        spec = whisper_spec.WhisperSpec(
            model.config.encoder_layers,
            model.config.encoder_attention_heads,
        )

        self.set_encoder(spec.encoder, model.model.encoder)
        self.set_decoder(spec.decoder, model.model.decoder)
        self.set_linear(spec.decoder.projection, model.proj_out)

        return spec

    def set_config(self, config, model, tokenizer):
        config.suppress_ids = model.config.suppress_tokens
        config.suppress_ids_begin = model.config.begin_suppress_tokens
        config.lang_ids = tokenizer.additional_special_tokens_ids[2:-6]

        config.alignment_heads = _WHISPER_ALIGNMENT_HEADS.get(model.name_or_path)
        if config.alignment_heads is None:
            # Use the last half layers for alignment by default.
            num_layers = model.config.decoder_layers
            num_heads = model.config.decoder_attention_heads
            config.alignment_heads = list(
                itertools.product(
                    range(num_layers // 2, num_layers),
                    range(num_heads),
                )
            )

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)

        # Add timestamp tokens.
        tokens.extend(
            "<|%.2f|>" % (i * 0.02)
            for i in range(model.config.vocab_size - len(tokens))
        )

        return tokens

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_encoder(self, spec, encoder):
        self.set_conv1d(spec.conv1, encoder.conv1)
        self.set_conv1d(spec.conv2, encoder.conv2)
        super().set_encoder(spec, encoder)

    def set_decoder(self, spec, decoder):
        self.set_embeddings(spec.embeddings, decoder.embed_tokens)
        super().set_decoder(spec, decoder)

    def set_common_layers(self, spec, module):
        self.set_position_encodings(spec.position_encodings, module.embed_positions)
        self.set_layer_norm(spec.layer_norm, module.layer_norm)

    def set_conv1d(self, spec, module):
        spec.weight = module.weight.numpy()
        spec.bias = module.bias.numpy()


@register_loader("T5Config")
class T5Loader(ModelLoader):
    @property
    def architecture_name(self):
        return "T5ForConditionalGeneration"

    def get_model_spec(self, model):
        spec = transformer_spec.TransformerSpec.from_config(
            (model.config.num_layers, model.config.num_decoder_layers),
            model.config.num_heads,
            pre_norm=True,
            activation=_SUPPORTED_ACTIVATIONS[model.config.dense_act_fn],
            ffn_glu=model.config.is_gated_act,
            relative_attention_bias=True,
            rms_norm=True,
        )

        self.set_stack(spec.encoder, model.encoder)
        self.set_stack(spec.decoder, model.decoder, is_decoder=True)
        self.set_linear(spec.decoder.projection, model.lm_head)

        if model.config.tie_word_embeddings:
            spec.decoder.scale_outputs = model.config.d_model**-0.5

        return spec

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)

        extra_ids = model.config.vocab_size - len(tokens)
        for i in range(extra_ids):
            tokens.append("<extra_id_%d>" % i)

        return tokens

    def set_vocabulary(self, spec, tokens):
        spec.register_source_vocabulary(tokens)
        spec.register_target_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.pad_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token
        config.decoder_start_token = tokenizer.pad_token

    def set_stack(self, spec, module, is_decoder=False):
        self.set_layer_norm(spec.layer_norm, module.final_layer_norm)
        self.set_embeddings(
            spec.embeddings[0]
            if isinstance(spec.embeddings, list)
            else spec.embeddings,
            module.embed_tokens,
        )

        spec.scale_embeddings = False

        for i, (layer_spec, block) in enumerate(zip(spec.layer, module.block)):
            self.set_self_attention(layer_spec.self_attention, block.layer[0])

            if i > 0:
                # Reuse relative attention bias from the first layer.
                first_self_attention = spec.layer[0].self_attention
                layer_spec.self_attention.relative_attention_bias = (
                    first_self_attention.relative_attention_bias
                )
                layer_spec.self_attention.relative_attention_max_distance = (
                    first_self_attention.relative_attention_max_distance
                )

            if is_decoder:
                self.set_cross_attention(layer_spec.attention, block.layer[1])

            self.set_ffn(layer_spec.ffn, block.layer[-1])

    def set_ffn(self, spec, module):
        if hasattr(spec, "linear_0_noact"):
            self.set_linear(spec.linear_0, module.DenseReluDense.wi_0)
            self.set_linear(spec.linear_0_noact, module.DenseReluDense.wi_1)
        else:
            self.set_linear(spec.linear_0, module.DenseReluDense.wi)

        self.set_linear(spec.linear_1, module.DenseReluDense.wo)
        self.set_layer_norm(spec.layer_norm, module.layer_norm)

    def set_self_attention(self, spec, module):
        self.set_attention(spec, module.SelfAttention, self_attention=True)
        self.set_layer_norm(spec.layer_norm, module.layer_norm)

    def set_cross_attention(self, spec, module):
        self.set_attention(spec, module.EncDecAttention)
        self.set_layer_norm(spec.layer_norm, module.layer_norm)

    def set_attention(self, spec, attention, self_attention=False):
        spec.queries_scale = 1.0

        split_layers = [common_spec.LinearSpec() for _ in range(3)]
        self.set_linear(split_layers[0], attention.q)
        self.set_linear(split_layers[1], attention.k)
        self.set_linear(split_layers[2], attention.v)

        if self_attention:
            utils.fuse_linear(spec.linear[0], split_layers)
        else:
            utils.fuse_linear(spec.linear[0], split_layers[:1])
            utils.fuse_linear(spec.linear[1], split_layers[1:])

        self.set_linear(spec.linear[-1], attention.o)

        if attention.has_relative_attention_bias:
            spec.relative_attention_bias = (
                attention.relative_attention_bias.weight.numpy()
            )
            spec.relative_attention_max_distance = np.dtype("int32").type(
                attention.relative_attention_max_distance
            )

    def set_layer_norm(self, spec, layer_norm):
        spec.gamma = layer_norm.weight.numpy()


@register_loader("MT5Config")
class MT5Loader(T5Loader):
    @property
    def architecture_name(self):
        return "MT5ForConditionalGeneration"


@register_loader("BloomConfig")
class BloomLoader(ModelLoader):
    @property
    def architecture_name(self):
        return "BloomForCausalLM"

    def get_model_spec(self, model):
        spec = transformer_spec.TransformerDecoderModelSpec.from_config(
            model.config.n_layer,
            model.config.n_head,
            pre_norm=True,
            activation=common_spec.Activation.GELUTanh,
            layernorm_embedding=True,
            alibi=True,
        )

        self.set_decoder(spec.decoder, model.transformer)
        self.set_linear(spec.decoder.projection, model.lm_head)
        return spec

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)

        extra_ids = model.config.vocab_size - len(tokens)
        for i in range(extra_ids):
            tokens.append("<extra_id_%d>" % i)

        return tokens

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.bos_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token

    def set_decoder(self, spec, module):
        spec.scale_embeddings = False
        self.set_embeddings(spec.embeddings, module.word_embeddings)
        self.set_layer_norm(spec.layernorm_embedding, module.word_embeddings_layernorm)
        self.set_layer_norm(spec.layer_norm, module.ln_f)

        for layer_spec, layer in zip(spec.layer, module.h):
            self.set_layer_norm(
                layer_spec.self_attention.layer_norm, layer.input_layernorm
            )
            self.set_qkv_linear(
                layer_spec.self_attention.linear[0],
                layer.self_attention.query_key_value,
                layer.self_attention.num_heads,
            )
            self.set_linear(
                layer_spec.self_attention.linear[1], layer.self_attention.dense
            )

            self.set_layer_norm(
                layer_spec.ffn.layer_norm, layer.post_attention_layernorm
            )
            self.set_linear(layer_spec.ffn.linear_0, layer.mlp.dense_h_to_4h)
            self.set_linear(layer_spec.ffn.linear_1, layer.mlp.dense_4h_to_h)

    def set_qkv_linear(self, spec, module, num_heads):
        weight = module.weight
        weight = weight.reshape(num_heads, 3, -1, weight.shape[-1])
        weight = weight.transpose(0, 1)
        weight = weight.reshape(-1, weight.shape[-1])

        bias = module.bias
        bias = bias.reshape(num_heads, 3, -1)
        bias = bias.transpose(0, 1)
        bias = bias.reshape(-1)

        spec.weight = weight.numpy()
        spec.bias = bias.numpy()


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
    parser.add_argument(
        "--activation_scales",
        help=(
            "Path to the pre-computed activation scales. Models may "
            "use them to rescale some weights to smooth the intermediate activations "
            "and improve the quantization accuracy. See "
            "https://github.com/mit-han-lab/smoothquant."
        ),
    )
    parser.add_argument(
        "--copy_files",
        nargs="+",
        help=(
            "List of filenames to copy from the Hugging Face model to the converted "
            "model directory."
        ),
    )

    Converter.declare_arguments(parser)
    args = parser.parse_args()
    converter = TransformersConverter(
        args.model,
        activation_scales=args.activation_scales,
        copy_files=args.copy_files,
        load_as_float16=args.quantization == "float16",
    )
    converter.convert_from_args(args)


if __name__ == "__main__":
    main()


# Cross-attention heads that are highly correlated to the word-level timing,
# i.e. the alignment between audio and text tokens.
# Obtained from https://github.com/openai/whisper/blob/v20230306/whisper/__init__.py#L31-L45
_WHISPER_ALIGNMENT_HEADS = {
    "openai/whisper-tiny.en": [
        (1, 0),
        (2, 0),
        (2, 5),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
        (3, 4),
    ],
    "openai/whisper-tiny": [(2, 2), (3, 0), (3, 2), (3, 3), (3, 4), (3, 5)],
    "openai/whisper-base.en": [(3, 3), (4, 7), (5, 1), (5, 5), (5, 7)],
    "openai/whisper-base": [
        (3, 1),
        (4, 2),
        (4, 3),
        (4, 7),
        (5, 1),
        (5, 2),
        (5, 4),
        (5, 6),
    ],
    "openai/whisper-small.en": [
        (6, 6),
        (7, 0),
        (7, 3),
        (7, 8),
        (8, 2),
        (8, 5),
        (8, 7),
        (9, 0),
        (9, 4),
        (9, 8),
        (9, 10),
        (10, 0),
        (10, 1),
        (10, 2),
        (10, 3),
        (10, 6),
        (10, 11),
        (11, 2),
        (11, 4),
    ],
    "openai/whisper-small": [
        (5, 3),
        (5, 9),
        (8, 0),
        (8, 4),
        (8, 7),
        (8, 8),
        (9, 0),
        (9, 7),
        (9, 9),
        (10, 5),
    ],
    "openai/whisper-medium.en": [
        (11, 4),
        (14, 1),
        (14, 12),
        (14, 14),
        (15, 4),
        (16, 0),
        (16, 4),
        (16, 9),
        (17, 12),
        (17, 14),
        (18, 7),
        (18, 10),
        (18, 15),
        (20, 0),
        (20, 3),
        (20, 9),
        (20, 14),
        (21, 12),
    ],
    "openai/whisper-medium": [(13, 15), (15, 4), (15, 15), (16, 1), (20, 0), (23, 4)],
    "openai/whisper-large": [
        (9, 19),
        (11, 2),
        (11, 4),
        (11, 17),
        (22, 7),
        (22, 11),
        (22, 17),
        (23, 2),
        (23, 15),
    ],
    "openai/whisper-large-v2": [
        (10, 12),
        (13, 17),
        (16, 11),
        (16, 12),
        (16, 13),
        (17, 15),
        (17, 16),
        (18, 4),
        (18, 11),
        (18, 19),
        (19, 11),
        (21, 2),
        (21, 3),
        (22, 3),
        (22, 9),
        (22, 12),
        (23, 5),
        (23, 7),
        (23, 13),
        (25, 5),
        (26, 1),
        (26, 12),
        (27, 15),
    ],
}
