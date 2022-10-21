import os

import test_utils

import ctranslate2


def test_marian_model_conversion(tmpdir):
    model_dir = os.path.join(test_utils.get_data_dir(), "models", "opus-mt-ende")
    converter = ctranslate2.converters.OpusMTConverter(model_dir)
    output_dir = str(tmpdir.join("ctranslate2_model"))
    converter.convert(output_dir)
    translator = ctranslate2.Translator(output_dir)
    output = translator.translate_batch([["▁Hello", "▁world", "!"]])
    assert output[0].hypotheses[0] == ["▁Hallo", "▁Welt", "!"]
