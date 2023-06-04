import os

import test_utils

import ctranslate2

from ctranslate2.converters.marian import load_vocab


def test_marian_model_conversion(tmp_dir):
    model_dir = os.path.join(test_utils.get_data_dir(), "models", "opus-mt-ende")
    converter = ctranslate2.converters.OpusMTConverter(model_dir)
    output_dir = str(tmp_dir.join("ctranslate2_model"))
    converter.convert(output_dir)
    translator = ctranslate2.Translator(output_dir)
    output = translator.translate_batch([["▁Hello", "▁world", "!"]])
    assert output[0].hypotheses[0] == ["▁Hallo", "▁Welt", "!"]


def test_marian_vocab_parsing():
    vocab_path = os.path.join(test_utils.get_data_dir(), "marian", "vocab.yml")
    expected_path = os.path.join(
        test_utils.get_data_dir(), "marian", "vocab.expected.txt"
    )

    with open(expected_path, encoding="utf-8") as expected_file:
        expected_tokens = [line.rstrip("\n") for line in expected_file]

    assert load_vocab(vocab_path) == expected_tokens
