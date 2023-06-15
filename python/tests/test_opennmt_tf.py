import copy
import os

import opennmt
import pytest
import tensorflow as tf
import test_utils

import ctranslate2

from ctranslate2.converters import opennmt_tf


@pytest.mark.parametrize("model_path", ["v1/checkpoint", "v2/checkpoint"])
def test_opennmt_tf_model_conversion(tmp_dir, model_path):
    model_path = os.path.join(
        test_utils.get_data_dir(),
        "models",
        "transliteration-aren-all",
        "opennmt_tf",
        model_path,
    )

    config = {
        "model_dir": model_path,
        "data": {
            "source_vocabulary": os.path.join(model_path, "ar.vocab"),
            "target_vocabulary": os.path.join(model_path, "en.vocab"),
        },
    }

    original_config = copy.deepcopy(config)

    converter = ctranslate2.converters.OpenNMTTFConverter.from_config(config)

    # auto_config should not update the configuration in place.
    assert config == original_config

    output_dir = str(tmp_dir.join("ctranslate2_model"))
    converter.convert(output_dir)

    translator = ctranslate2.Translator(output_dir)
    output = translator.translate_batch([["آ", "ت", "ز", "م", "و", "ن"]])
    assert output[0].hypotheses[0] == ["a", "t", "z", "m", "o", "n"]


@pytest.mark.parametrize("quantization", ["float16", "int16", "int8", "int8_float16"])
def test_opennmt_tf_model_quantization(tmp_dir, quantization):
    model_path = os.path.join(
        test_utils.get_data_dir(),
        "models",
        "transliteration-aren-all",
        "opennmt_tf",
        "v2",
        "checkpoint",
    )

    config = {
        "model_dir": model_path,
        "data": {
            "source_vocabulary": os.path.join(model_path, "ar.vocab"),
            "target_vocabulary": os.path.join(model_path, "en.vocab"),
        },
    }

    converter = ctranslate2.converters.OpenNMTTFConverter.from_config(config)
    output_dir = str(tmp_dir.join("ctranslate2_model"))
    converter.convert(output_dir, quantization=quantization)
    translator = ctranslate2.Translator(output_dir)
    output = translator.translate_batch([["آ", "ت", "ز", "م", "و", "ن"]])
    assert output[0].hypotheses[0] == ["a", "t", "z", "m", "o", "n"]


def test_opennmt_tf_model_conversion_invalid_vocab():
    model_path = os.path.join(
        test_utils.get_data_dir(),
        "models",
        "transliteration-aren-all",
        "opennmt_tf",
        "v2",
        "checkpoint",
    )

    # Swap source and target vocabularies.
    config = {
        "model_dir": model_path,
        "data": {
            "source_vocabulary": os.path.join(model_path, "en.vocab"),
            "target_vocabulary": os.path.join(model_path, "ar.vocab"),
        },
    }

    with pytest.raises(ValueError, match="not compatible"):
        ctranslate2.converters.OpenNMTTFConverter.from_config(config)


def _create_vocab(tmp_dir, name="vocab", size=10):
    vocab = opennmt.data.Vocab()
    for i in range(size):
        vocab.add(str(i))
    vocab_path = str(tmp_dir.join("%s.txt" % name))
    vocab.serialize(vocab_path)
    return vocab_path


def test_opennmt_tf_model_conversion_invalid_dir(tmp_dir):
    model_path = str(tmp_dir.join("model").ensure(dir=1))
    vocab_path = _create_vocab(tmp_dir)
    config = {
        "model_dir": model_path,
        "data": {"source_vocabulary": vocab_path, "target_vocabulary": vocab_path},
    }
    with pytest.raises(RuntimeError, match="checkpoint"):
        ctranslate2.converters.OpenNMTTFConverter.from_config(
            config, model="TransformerBase"
        )


def test_opennmt_tf_shared_embeddings_conversion(tmp_dir):
    # Issue https://github.com/OpenNMT/CTranslate2/issues/118
    model = opennmt.models.Transformer(
        opennmt.inputters.WordEmbedder(32),
        opennmt.inputters.WordEmbedder(32),
        num_layers=3,
        num_units=32,
        num_heads=4,
        ffn_inner_dim=64,
        share_embeddings=opennmt.models.EmbeddingsSharingLevel.ALL,
    )

    vocab_path = _create_vocab(tmp_dir)
    model.initialize({"source_vocabulary": vocab_path, "target_vocabulary": vocab_path})
    model.create_variables()

    converter = ctranslate2.converters.OpenNMTTFConverter(model)
    output_dir = str(tmp_dir.join("ctranslate2_model"))
    converter.convert(output_dir)

    assert os.path.isfile(os.path.join(output_dir, "shared_vocabulary.json"))

    # Check that the translation runs.
    translator = ctranslate2.Translator(output_dir)
    translator.translate_batch([["1", "2", "3"]], max_decoding_length=10)


@pytest.mark.parametrize("encoder_only", [True, False])
def test_opennmt_tf_postnorm_transformer_conversion(tmp_dir, encoder_only):
    model = opennmt.models.Transformer(
        opennmt.inputters.WordEmbedder(32),
        opennmt.inputters.WordEmbedder(32),
        num_layers=3,
        num_units=32,
        num_heads=4,
        ffn_inner_dim=64,
        pre_norm=encoder_only,
    )

    if encoder_only:
        model.encoder = opennmt.encoders.SelfAttentionEncoder(
            num_layers=3,
            num_units=32,
            num_heads=4,
            ffn_inner_dim=64,
            pre_norm=False,
        )

    vocab_path = _create_vocab(tmp_dir)
    model.initialize({"source_vocabulary": vocab_path, "target_vocabulary": vocab_path})
    model.create_variables()

    converter = ctranslate2.converters.OpenNMTTFConverter(model)
    output_dir = str(tmp_dir.join("ctranslate2_model"))
    converter.convert(output_dir)


def test_opennmt_tf_gpt_conversion(tmp_dir):
    vocabulary = _create_vocab(tmp_dir, "vocab")
    model = opennmt.models.GPT2Small()
    model.initialize(dict(vocabulary=vocabulary))
    model.create_variables()

    output_dir = str(tmp_dir.join("ctranslate2_model"))
    converter = ctranslate2.converters.OpenNMTTFConverter(model)
    converter.convert(output_dir)

    assert os.path.isfile(os.path.join(output_dir, "vocabulary.json"))


def test_opennmt_tf_multi_features(tmp_dir):
    model = opennmt.models.Transformer(
        opennmt.inputters.ParallelInputter(
            [
                opennmt.inputters.WordEmbedder(24),
                opennmt.inputters.WordEmbedder(8),
            ],
            reducer=opennmt.layers.ConcatReducer(),
        ),
        opennmt.inputters.WordEmbedder(32),
        num_layers=3,
        num_units=32,
        num_heads=4,
        ffn_inner_dim=64,
    )

    model.initialize(
        {
            "source_1_vocabulary": _create_vocab(tmp_dir, "source_1", 50),
            "source_2_vocabulary": _create_vocab(tmp_dir, "source_2", 10),
            "target_vocabulary": _create_vocab(tmp_dir, "target", 60),
        }
    )
    model.create_variables()

    converter = ctranslate2.converters.OpenNMTTFConverter(model)
    output_dir = str(tmp_dir.join("ctranslate2_model"))
    converter.convert(output_dir)

    assert os.path.isfile(os.path.join(output_dir, "source_1_vocabulary.json"))
    assert os.path.isfile(os.path.join(output_dir, "source_2_vocabulary.json"))
