# -*- coding: utf-8 -*-

import os
import pytest

from ctranslate2 import Translator
from ctranslate2.converters.opennmt_tf import OpenNMTTFConverter
from ctranslate2.specs.transformer_spec import TransformerSpec


_TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "..", "..", "tests", "data")


def _get_transliterator():
    model_path = os.path.join(_TEST_DATA_DIR, "models", "v2", "aren-transliteration")
    return Translator(model_path)


def test_invalid_model_path():
    with pytest.raises(RuntimeError):
        Translator("xxx")

def test_batch_translation():
    translator = _get_transliterator()
    output = translator.translate_batch([
        ["آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"],
        ["آ" ,"ت" ,"ش" ,"ي" ,"س" ,"و" ,"ن"]])
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
    translator.translate_file(input_path, output_path, max_batch_size=32)
    with open(output_path) as output_file:
        lines = output_file.readlines()
        assert lines[0].strip() == "a t z m o n"
        assert lines[1].strip() == "a c h i s o n"

def test_empty_translation():
    translator = _get_transliterator()
    assert translator.translate_batch([]) == []
    assert translator.translate_batch(None) == []

def test_target_prefix():
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"]], target_prefix=[["a", "t", "s"]])
    assert output[0][0]["tokens"][:3] == ["a", "t", "s"]

def test_num_hypotheses():
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"]], beam_size=4, num_hypotheses=2)
    assert len(output[0]) == 2

def test_max_decoding_length():
    translator = _get_transliterator()
    output = translator.translate_batch([["آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"]], max_decoding_length=2)
    assert output[0][0]["tokens"] == ["a", "t"]

def test_min_decoding_length():
    translator = _get_transliterator()
    output = translator.translate_batch([["آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"]], min_decoding_length=7)
    assert len(output[0][0]["tokens"]) > 6  # 6 is the expected target length.

def test_return_attention():
    translator = _get_transliterator()
    output = translator.translate_batch([["آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"]], return_attention=True)
    attention = output[0][0]["attention"]
    assert len(attention) == 6  # Target length.
    assert len(attention[0]) == 6  # Source length.

@pytest.mark.skipif(
    not os.path.isdir(os.path.join(_TEST_DATA_DIR, "models", "transliteration-aren-all")),
    reason="Data files are not available")
@pytest.mark.parametrize(
    "model_path,src_vocab,tgt_vocab",
    [("v1/savedmodel", None, None),
     ("v1/checkpoint", "ar.vocab", "en.vocab")])
def test_opennmt_tf_model_conversion(tmpdir, model_path, src_vocab, tgt_vocab):
    model_path = os.path.join(_TEST_DATA_DIR, "models", "transliteration-aren-all", model_path)
    if src_vocab is not None:
        src_vocab = os.path.join(model_path, src_vocab)
    if tgt_vocab is not None:
        tgt_vocab = os.path.join(model_path, tgt_vocab)
    converter = OpenNMTTFConverter(model_path, src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    output_dir = str(tmpdir.join("ctranslate2_model"))
    converter.convert(output_dir, "TransformerBase")
    translator = Translator(output_dir)
    output = translator.translate_batch([["آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"]])
    assert output[0][0]["tokens"] == ["a", "t", "z", "m", "o", "n"]
