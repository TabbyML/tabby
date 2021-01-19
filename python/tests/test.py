# -*- coding: utf-8 -*-

import os
import pytest
import numpy as np

import ctranslate2

from ctranslate2.specs.model_spec import OPTIONAL, index_spec
from ctranslate2.specs import transformer_spec
from ctranslate2.converters import opennmt_tf


_TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "tests", "data"
)


def _get_model_path():
    return os.path.join(_TEST_DATA_DIR, "models", "v2", "aren-transliteration")


def _get_transliterator():
    return ctranslate2.Translator(_get_model_path())


def test_invalid_model_path():
    with pytest.raises(RuntimeError):
        ctranslate2.Translator("xxx")


def test_contains_model(tmpdir):
    assert ctranslate2.contains_model(_get_model_path())

    model_dir = tmpdir.join("model")
    model_dir.ensure(dir=1)
    assert not ctranslate2.contains_model(str(model_dir))
    model_dir.join("model.bin").ensure(file=1)
    assert ctranslate2.contains_model(str(model_dir))


def test_translator_properties():
    translator = ctranslate2.Translator(_get_model_path(), inter_threads=2)
    assert translator.model_is_loaded
    assert translator.device == "cpu"
    assert translator.device_index == 0
    assert translator.num_translators == 2
    assert translator.num_queued_batches == 0


def test_compute_type():
    model_path = _get_model_path()
    with pytest.raises(ValueError):
        ctranslate2.Translator(model_path, compute_type="float64")
    with pytest.raises(TypeError):
        ctranslate2.Translator(model_path, compute_type=["int8", "int16"])
    ctranslate2.Translator(model_path, compute_type="int8")
    ctranslate2.Translator(model_path, compute_type={"cuda": "float16", "cpu": "int8"})


@pytest.mark.parametrize("max_batch_size", [0, 1])
def test_batch_translation(max_batch_size):
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"], ["آ", "ت", "ش", "ي", "س", "و", "ن"]],
        max_batch_size=max_batch_size,
    )
    assert len(output) == 2
    assert len(output[0]) == 1  # One hypothesis.
    assert len(output[1]) == 1
    assert output[0][0]["tokens"] == ["a", "t", "z", "m", "o", "n"]
    assert output[0][0]["score"] < 0
    assert "attention" not in output[0][0]
    assert output[1][0]["tokens"] == ["a", "c", "h", "i", "s", "o", "n"]


def test_file_translation(tmpdir):
    input_path = str(tmpdir.join("input.txt"))
    output_path = str(tmpdir.join("output.txt"))
    with open(input_path, "w") as input_file:
        input_file.write("آ ت ز م و ن")
        input_file.write("\n")
        input_file.write("آ ت ش ي س و ن")
        input_file.write("\n")
    translator = _get_transliterator()
    stats = translator.translate_file(input_path, output_path, max_batch_size=32)
    with open(output_path) as output_file:
        lines = output_file.readlines()
        assert lines[0].strip() == "a t z m o n"
        assert lines[1].strip() == "a c h i s o n"
    assert stats[0] == 13  # Number of generated target tokens.
    assert stats[1] == 2  # Number of translated examples.
    assert isinstance(stats[2], float)  # Total time in milliseconds.


def test_raw_file_translation(tmpdir):
    input_path = str(tmpdir.join("input.txt"))
    output_path = str(tmpdir.join("output.txt"))
    with open(input_path, "w") as input_file:
        input_file.write("آتزمون")
        input_file.write("\n")
        input_file.write("آتشيسون")
        input_file.write("\n")

    translator = ctranslate2.Translator(_get_model_path())
    tokenize_fn = lambda text: list(text)
    detokenize_fn = lambda tokens: "".join(tokens)
    max_batch_size = 4

    with pytest.raises(ValueError):
        translator.translate_file(
            input_path, output_path, max_batch_size, tokenize_fn=tokenize_fn
        )
    with pytest.raises(ValueError):
        translator.translate_file(
            input_path, output_path, max_batch_size, detokenize_fn=detokenize_fn
        )

    translator.translate_file(
        input_path,
        output_path,
        max_batch_size,
        tokenize_fn=tokenize_fn,
        detokenize_fn=detokenize_fn,
    )

    with open(output_path) as output_file:
        lines = output_file.readlines()
        assert lines[0].strip() == "atzmon"
        assert lines[1].strip() == "achison"


def test_file_translation_with_prefix(tmpdir):
    source_path = str(tmpdir.join("input.txt"))
    target_path = str(tmpdir.join("target.txt"))
    output_path = str(tmpdir.join("output.txt"))
    with open(source_path, "w") as source_file:
        source_file.write("آ ت ز م و ن")
        source_file.write("\n")
        source_file.write("آ ت ش ي س و ن")
        source_file.write("\n")
    with open(target_path, "w") as target_file:
        target_file.write("a t s\n")

    translator = _get_transliterator()
    max_batch_size = 4

    with pytest.raises(RuntimeError):
        # One line is missing from target_path.
        translator.translate_file(
            source_path, output_path, max_batch_size, target_path=target_path
        )

    with open(target_path, "a") as target_file:
        target_file.write("\n")  # No prefix.

    translator.translate_file(
        source_path, output_path, max_batch_size, target_path=target_path
    )

    with open(output_path) as output_file:
        lines = output_file.readlines()
        assert lines[0].strip() == "a t s u m o n"
        assert lines[1].strip() == "a c h i s o n"


def test_raw_file_translation_with_prefix(tmpdir):
    source_path = str(tmpdir.join("input.txt"))
    target_path = str(tmpdir.join("target.txt"))
    output_path = str(tmpdir.join("output.txt"))
    with open(source_path, "w") as source_file:
        source_file.write("آتزمون")
        source_file.write("\n")
        source_file.write("آتشيسون")
        source_file.write("\n")
    with open(target_path, "w") as target_file:
        # Write target in reverse to use a different tokenization.
        target_file.write("sta\n")
        target_file.write("\n")

    translator = ctranslate2.Translator(_get_model_path())
    source_tokenize_fn = lambda text: list(text)
    target_tokenize_fn = lambda text: list(reversed(list(text)))
    detokenize_fn = lambda tokens: "".join(tokens)
    max_batch_size = 4

    with pytest.raises(ValueError):
        # Target tokenization is missing.
        translator.translate_file(
            source_path,
            output_path,
            max_batch_size,
            tokenize_fn=source_tokenize_fn,
            detokenize_fn=detokenize_fn,
            target_path=target_path,
        )

    translator.translate_file(
        source_path,
        output_path,
        max_batch_size,
        tokenize_fn=source_tokenize_fn,
        detokenize_fn=detokenize_fn,
        target_path=target_path,
        target_tokenize_fn=target_tokenize_fn,
    )

    with open(output_path) as output_file:
        lines = output_file.readlines()
        assert lines[0].strip() == "atsumon"
        assert lines[1].strip() == "achison"


def test_empty_translation():
    translator = _get_transliterator()
    assert translator.translate_batch([]) == []


def test_invalid_translation_options():
    translator = _get_transliterator()
    with pytest.raises(ValueError):
        translator.translate_batch(
            [["آ", "ت", "ز", "م", "و", "ن"]],
            min_decoding_length=10,
            max_decoding_length=5,
        )


def test_target_prefix():
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"], ["آ", "ت", "ش", "ي", "س", "و", "ن"]],
        target_prefix=[["a", "t", "s"], None],
    )
    assert output[0][0]["tokens"][:3] == ["a", "t", "s"]
    assert output[1][0]["tokens"] == ["a", "c", "h", "i", "s", "o", "n"]


def test_num_hypotheses():
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"]], beam_size=4, num_hypotheses=2
    )
    assert len(output[0]) == 2


def test_max_decoding_length():
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"]], max_decoding_length=2
    )
    assert output[0][0]["tokens"] == ["a", "t"]


def test_min_decoding_length():
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"]], min_decoding_length=7
    )
    assert len(output[0][0]["tokens"]) > 6  # 6 is the expected target length.


def test_return_attention():
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"]], return_attention=True
    )
    attention = output[0][0]["attention"]
    assert len(attention) == 6  # Target length.
    assert len(attention[0]) == 6  # Source length.


def test_ignore_scores():
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"]], beam_size=1, return_scores=False
    )
    assert "scores" not in output[0][0]


def test_return_alternatives():
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"]],
        target_prefix=[["a", "t"]],
        num_hypotheses=10,
        return_alternatives=True,
    )
    assert len(output[0]) == 10
    assert output[0][0]["tokens"] == ["a", "t", "z", "m", "o", "n"]
    assert output[0][1]["tokens"] == ["a", "t", "s", "u", "m", "o", "n"]


@pytest.mark.parametrize("to_cpu", [False, True])
def test_model_unload(to_cpu):
    batch = [["آ", "ت", "ز", "م", "و", "ن"]]
    translator = _get_transliterator()
    translator.unload_model(to_cpu=to_cpu)
    if not to_cpu:
        with pytest.raises(RuntimeError, match="unloaded"):
            translator.translate_batch(batch)
    translator.load_model()
    output = translator.translate_batch(batch)
    assert len(output) == 1
    assert output[0][0]["tokens"] == ["a", "t", "z", "m", "o", "n"]


_FRAMEWORK_DATA_EXIST = os.path.isdir(
    os.path.join(_TEST_DATA_DIR, "models", "transliteration-aren-all")
)


@pytest.mark.skipif(not _FRAMEWORK_DATA_EXIST, reason="Data files are not available")
@pytest.mark.parametrize(
    "model_path,src_vocab,tgt_vocab,model_spec",
    [
        ("v2/savedmodel", None, None, "TransformerBase"),
        (
            "v2/savedmodel",
            None,
            None,
            ctranslate2.specs.TransformerSpec(num_layers=6, num_heads=8),
        ),
        ("v1/checkpoint", "ar.vocab", "en.vocab", ctranslate2.specs.TransformerBase()),
        ("v2/checkpoint", "ar.vocab", "en.vocab", ctranslate2.specs.TransformerBase()),
    ],
)
def test_opennmt_tf_model_conversion(
    tmpdir, model_path, src_vocab, tgt_vocab, model_spec
):
    model_path = os.path.join(
        _TEST_DATA_DIR, "models", "transliteration-aren-all", "opennmt_tf", model_path
    )
    if src_vocab is not None:
        src_vocab = os.path.join(model_path, src_vocab)
    if tgt_vocab is not None:
        tgt_vocab = os.path.join(model_path, tgt_vocab)
    converter = ctranslate2.converters.OpenNMTTFConverter(
        model_path, src_vocab=src_vocab, tgt_vocab=tgt_vocab
    )
    output_dir = str(tmpdir.join("ctranslate2_model"))
    converter.convert(output_dir, model_spec)
    translator = ctranslate2.Translator(output_dir)
    output = translator.translate_batch([["آ", "ت", "ز", "م", "و", "ن"]])
    assert output[0][0]["tokens"] == ["a", "t", "z", "m", "o", "n"]


@pytest.mark.skipif(not _FRAMEWORK_DATA_EXIST, reason="Data files are not available")
@pytest.mark.parametrize("quantization", ["float16", "int16", "int8"])
def test_opennmt_tf_model_quantization(tmpdir, quantization):
    model_path = os.path.join(
        _TEST_DATA_DIR,
        "models",
        "transliteration-aren-all",
        "opennmt_tf",
        "v2",
        "checkpoint",
    )
    converter = ctranslate2.converters.OpenNMTTFConverter(
        model_path,
        src_vocab=os.path.join(model_path, "ar.vocab"),
        tgt_vocab=os.path.join(model_path, "en.vocab"),
    )
    output_dir = str(tmpdir.join("ctranslate2_model"))
    converter.convert(
        output_dir, ctranslate2.specs.TransformerBase(), quantization=quantization
    )
    translator = ctranslate2.Translator(output_dir)
    output = translator.translate_batch([["آ", "ت", "ز", "م", "و", "ن"]])
    assert output[0][0]["tokens"] == ["a", "t", "z", "m", "o", "n"]


@pytest.mark.skipif(not _FRAMEWORK_DATA_EXIST, reason="Data files are not available")
def test_opennmt_tf_variables_conversion(tmpdir):
    model_path = os.path.join(
        _TEST_DATA_DIR,
        "models",
        "transliteration-aren-all",
        "opennmt_tf",
        "v2",
        "checkpoint",
    )
    _, variables, src_vocab, tgt_vocab = opennmt_tf.load_model(
        model_path,
        src_vocab=os.path.join(model_path, "ar.vocab"),
        tgt_vocab=os.path.join(model_path, "en.vocab"),
    )
    converter = ctranslate2.converters.OpenNMTTFConverter(
        src_vocab=src_vocab, tgt_vocab=tgt_vocab, variables=variables
    )
    output_dir = str(tmpdir.join("ctranslate2_model"))
    converter.convert(output_dir, ctranslate2.specs.TransformerBase())
    translator = ctranslate2.Translator(output_dir)
    output = translator.translate_batch([["آ", "ت", "ز", "م", "و", "ن"]])
    assert output[0][0]["tokens"] == ["a", "t", "z", "m", "o", "n"]


@pytest.mark.skipif(not _FRAMEWORK_DATA_EXIST, reason="Data files are not available")
def test_opennmt_tf_model_conversion_invalid_vocab(tmpdir):
    model_path = os.path.join(
        _TEST_DATA_DIR,
        "models",
        "transliteration-aren-all",
        "opennmt_tf",
        "v2",
        "checkpoint",
    )
    # Swap source and target vocabularies.
    converter = ctranslate2.converters.OpenNMTTFConverter(
        model_path,
        src_vocab=os.path.join(model_path, "en.vocab"),
        tgt_vocab=os.path.join(model_path, "ar.vocab"),
    )
    output_dir = str(tmpdir.join("ctranslate2_model"))
    with pytest.raises(ValueError):
        converter.convert(output_dir, ctranslate2.specs.TransformerBase())


def test_opennmt_tf_shared_embeddings_conversion(tmpdir):
    # Issue https://github.com/OpenNMT/CTranslate2/issues/118
    import tensorflow as tf
    import opennmt

    vocab = opennmt.data.Vocab()
    for i in range(10):
        vocab.add(str(i))
    vocab_path = str(tmpdir.join("vocab.txt"))
    vocab.serialize(vocab_path)

    num_layers = 3
    num_heads = 4
    model = opennmt.models.Transformer(
        opennmt.inputters.WordEmbedder(32),
        opennmt.inputters.WordEmbedder(32),
        num_layers,
        num_units=32,
        num_heads=num_heads,
        ffn_inner_dim=64,
        share_embeddings=opennmt.models.EmbeddingsSharingLevel.ALL,
    )
    model.initialize({"source_vocabulary": vocab_path, "target_vocabulary": vocab_path})
    model.create_variables()

    checkpoint_prefix = str(tmpdir.join("ckpt"))
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.write(checkpoint_prefix)

    converter = ctranslate2.converters.OpenNMTTFConverter(
        model_path=checkpoint_prefix, src_vocab=vocab_path, tgt_vocab=vocab_path
    )
    output_dir = str(tmpdir.join("ctranslate2_model"))
    converter.convert(
        output_dir, ctranslate2.specs.TransformerSpec(num_layers, num_heads)
    )

    # Check that the translation runs.
    translator = ctranslate2.Translator(output_dir)
    translator.translate_batch([["1", "2", "3"]], max_decoding_length=10)


@pytest.mark.skipif(not _FRAMEWORK_DATA_EXIST, reason="Data files are not available")
def test_opennmt_py_model_conversion(tmpdir):
    model_path = os.path.join(
        _TEST_DATA_DIR,
        "models",
        "transliteration-aren-all",
        "opennmt_py",
        "aren_7000.pt",
    )
    converter = ctranslate2.converters.OpenNMTPyConverter(model_path)
    output_dir = str(tmpdir.join("ctranslate2_model"))
    converter.convert(output_dir, ctranslate2.specs.TransformerBase())
    translator = ctranslate2.Translator(output_dir)
    output = translator.translate_batch([["آ", "ت", "ز", "م", "و", "ن"]])
    assert output[0][0]["tokens"] == ["a", "t", "z", "m", "o", "n"]


@pytest.mark.skipif(not _FRAMEWORK_DATA_EXIST, reason="Data files are not available")
def test_opennmt_py_relative_transformer(tmpdir):
    model_path = os.path.join(
        _TEST_DATA_DIR,
        "models",
        "transliteration-aren-all",
        "opennmt_py",
        "aren_relative_6000.pt",
    )
    converter = ctranslate2.converters.OpenNMTPyConverter(model_path)
    output_dir = str(tmpdir.join("ctranslate2_model"))
    converter.convert(output_dir, ctranslate2.specs.TransformerBaseRelative())
    translator = ctranslate2.Translator(output_dir)
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"], ["آ", "ر", "ث", "ر"]]
    )
    assert output[0][0]["tokens"] == ["a", "t", "z", "o", "m", "o", "n"]
    assert output[1][0]["tokens"] == ["a", "r", "t", "h", "e", "r"]


def test_layer_spec_validate():
    class SubSpec(ctranslate2.specs.LayerSpec):
        def __init__(self):
            self.a = np.ones([5], dtype=np.float16)

    class Spec(ctranslate2.specs.LayerSpec):
        def __init__(self):
            self.a = np.zeros([5], dtype=np.float32)
            self.b = np.zeros([5], dtype=np.float16)
            self.c = np.zeros([5], dtype=np.int32)
            self.d = OPTIONAL
            self.e = SubSpec()
            self.f = True

    spec = Spec()
    spec.validate()
    assert spec.a.dtype == np.float32
    assert spec.b.dtype == np.float32
    assert spec.c.dtype == np.int32
    assert spec.d == OPTIONAL
    assert spec.e.a.dtype == np.float32
    assert spec.f.dtype == np.int8


def test_layer_spec_optimize():
    class SubSpec(ctranslate2.specs.LayerSpec):
        def __init__(self):
            self.a = np.ones([6], dtype=np.float32)

    class Spec(ctranslate2.specs.LayerSpec):
        def __init__(self):
            self.a = np.ones([5], dtype=np.float32)
            self.b = np.ones([5], dtype=np.float32)
            self.c = np.zeros([5], dtype=np.int32)
            self.d = np.dtype("float32").type(3.14)
            self.weight = np.ones([5, 4], dtype=np.float32)
            self.sub = SubSpec()

    spec = Spec()
    spec.optimize(quantization="int16")
    assert spec.a.dtype == np.float32
    assert spec.b == "a"
    assert spec.c.dtype == np.int32
    assert spec.d.dtype == np.float32
    assert spec.weight.dtype == np.int16
    assert spec.weight_scale.dtype == np.float32

    spec = Spec()
    spec.optimize(quantization="float16")
    assert spec.a.dtype == np.float16
    assert spec.b == "a"
    assert spec.c.dtype == np.int32
    assert spec.d.dtype == np.float32
    assert spec.weight.dtype == np.float16
    assert spec.sub.a.dtype == np.float16


def test_index_spec():
    spec = ctranslate2.specs.TransformerBase()
    assert isinstance(
        index_spec(spec, "encoder/layer_5"),
        transformer_spec.TransformerEncoderLayerSpec,
    )
    assert isinstance(
        index_spec(spec, "encoder/layer_5/ffn"), transformer_spec.FeedForwardSpec
    )
