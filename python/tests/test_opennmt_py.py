import os

import pytest
import test_utils

import ctranslate2


@test_utils.skip_on_windows
def test_opennmt_py_model_conversion(tmp_dir):
    model_path = os.path.join(
        test_utils.get_data_dir(),
        "models",
        "transliteration-aren-all",
        "opennmt_py",
        "aren_7000.pt",
    )
    converter = ctranslate2.converters.OpenNMTPyConverter(model_path)
    output_dir = str(tmp_dir.join("ctranslate2_model"))
    converter.convert(output_dir)
    translator = ctranslate2.Translator(output_dir)
    output = translator.translate_batch([["آ", "ت", "ز", "م", "و", "ن"]])
    assert output[0].hypotheses[0] == ["a", "t", "z", "u", "m", "o", "n"]


@test_utils.skip_on_windows
def test_opennmt_py_relative_transformer(tmp_dir):
    model_path = os.path.join(
        test_utils.get_data_dir(),
        "models",
        "transliteration-aren-all",
        "opennmt_py",
        "aren_relative_6000.pt",
    )
    converter = ctranslate2.converters.OpenNMTPyConverter(model_path)
    output_dir = str(tmp_dir.join("ctranslate2_model"))
    converter.convert(output_dir)
    translator = ctranslate2.Translator(output_dir)
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"], ["آ", "ر", "ث", "ر"]]
    )
    assert output[0].hypotheses[0] == ["a", "t", "z", "o", "m", "o", "n"]
    assert output[1].hypotheses[0] == ["a", "r", "t", "h", "e", "r"]


@test_utils.skip_on_windows
def test_opennmt_py_relative_transformer_return_alternatives(tmp_dir):
    # Test for issue https://github.com/OpenNMT/CTranslate2/issues/1394
    model_path = os.path.join(
        test_utils.get_data_dir(),
        "models",
        "transliteration-aren-all",
        "opennmt_py",
        "aren_relative_6000.pt",
    )
    converter = ctranslate2.converters.OpenNMTPyConverter(model_path)
    output_dir = str(tmp_dir.join("ctranslate2_model"))
    converter.convert(output_dir)
    translator = ctranslate2.Translator(output_dir)
    translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"]],
        target_prefix=[["a", "t"]],
        return_alternatives=True,
        beam_size=5,
        num_hypotheses=5,
    )


@test_utils.skip_on_windows
@pytest.mark.parametrize(
    "filename", ["aren_features_concat_10000.pt", "aren_features_sum_10000.pt"]
)
def test_opennmt_py_source_features(tmp_dir, filename):
    model_path = os.path.join(
        test_utils.get_data_dir(),
        "models",
        "transliteration-aren-all",
        "opennmt_py",
        filename,
    )
    converter = ctranslate2.converters.OpenNMTPyConverter(model_path)
    output_dir = str(tmp_dir.join("ctranslate2_model"))
    converter.convert(output_dir)
    assert os.path.isfile(os.path.join(output_dir, "source_1_vocabulary.json"))
    assert os.path.isfile(os.path.join(output_dir, "source_2_vocabulary.json"))

    source = [
        ["آ", "ت", "ز", "م", "و", "ن"],
        ["آ", "ت", "ش", "ي", "س", "و", "ن"],
    ]
    source_features = [
        ["0", "1", "2", "3", "4", "5"],
        ["0", "1", "2", "3", "4", "5", "6"],
    ]
    expected_target = [
        ["a", "t", "z", "m", "o", "n"],
        ["a", "c", "h", "i", "s", "o", "n"],
    ]

    source_w_features = []
    for tokens, features in zip(source, source_features):
        source_w_features.append(["%s￨%s" % pair for pair in zip(tokens, features)])

    translator = ctranslate2.Translator(output_dir)
    with pytest.raises(ValueError, match="features"):
        translator.translate_batch(source)

    outputs = translator.translate_batch(source_w_features)
    for output, expected_hypothesis in zip(outputs, expected_target):
        assert output.hypotheses[0] == expected_hypothesis

    input_path = str(tmp_dir.join("input.txt"))
    output_path = str(tmp_dir.join("output.txt"))

    test_utils.write_tokens(source, input_path)
    with pytest.raises(ValueError, match="features"):
        translator.translate_file(input_path, output_path)

    test_utils.write_tokens(source_w_features, input_path)
    translator.translate_file(input_path, output_path)
    with open(output_path) as output_file:
        for line, expected_hypothesis in zip(output_file, expected_target):
            assert line.strip().split() == expected_hypothesis


@test_utils.skip_on_windows
def test_opennmt_py_transformer_lm(tmp_dir):
    model_path = os.path.join(test_utils.get_data_dir(), "models", "pi_lm_step_5000.pt")
    if not os.path.exists(model_path):
        pytest.skip("Checkpoint file is not available")

    converter = ctranslate2.converters.OpenNMTPyConverter(model_path)
    output_dir = str(tmp_dir.join("ctranslate2_model"))
    converter.convert(output_dir)

    generator = ctranslate2.Generator(output_dir)
    results = generator.generate_batch([["<s>", "3", ".", "1", "4"]], max_length=12)

    assert "".join(results[0].sequences[0]) == "3.1415926535"
