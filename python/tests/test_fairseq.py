import os

import pytest
import test_utils

import ctranslate2


@test_utils.skip_on_windows
def test_fairseq_model_conversion(tmp_dir):
    data_dir = os.path.join(
        test_utils.get_data_dir(),
        "models",
        "transliteration-aren-all",
        "fairseq",
    )
    converter = ctranslate2.converters.FairseqConverter(
        os.path.join(data_dir, "model.pt"), data_dir
    )
    output_dir = str(tmp_dir.join("ctranslate2_model"))
    converter.convert(output_dir)
    translator = ctranslate2.Translator(output_dir)
    output = translator.translate_batch([["آ", "ت", "ز", "م", "و", "ن"]])
    assert output[0].hypotheses[0] == ["a", "t", "z", "m", "o", "n"]

    output = translator.score_batch(
        [["آ", "ت", "ز", "م", "و", "ن"]], [["a", "t", "z", "m", "o", "n"]]
    )

    expected_log_probs = [
        -0.036438941955566406,
        -0.20800018310546875,
        -0.06881046295166016,
        -0.5400338172912598,
        -0.12033462524414062,
        -0.1102609634399414,
        -0.044970035552978516,
    ]

    assert output[0].tokens == ["a", "t", "z", "m", "o", "n", "</s>"]
    assert output[0].log_probs == pytest.approx(expected_log_probs, 1e-5)


@test_utils.skip_on_windows
def test_fairseq_user_start_token(tmp_dir):
    data_dir = os.path.join(
        test_utils.get_data_dir(),
        "models",
        "transliteration-aren-all",
        "fairseq",
    )
    converter = ctranslate2.converters.FairseqConverter(
        os.path.join(data_dir, "model.pt"), data_dir, no_default_special_tokens=True
    )
    output_dir = str(tmp_dir.join("ctranslate2_model"))
    converter.convert(output_dir)
    translator = ctranslate2.Translator(output_dir)
    tokens = ["آ", "ت", "ز", "م", "و", "ن"]
    tokens += ["</s>"]

    with pytest.raises(ValueError, match="start token"):
        translator.translate_batch([tokens])

    output = translator.translate_batch([tokens], target_prefix=[["</s>"]])
    assert output[0].hypotheses[0] == ["a", "t", "z", "m", "o", "n"]

    output = translator.translate_batch([["</s>"]], target_prefix=[["</s>"]])
    assert output[0].hypotheses[0] == []

    output = translator.translate_batch(
        [["</s>"]],
        target_prefix=[["</s>", "oovtoken"]],
        return_scores=True,
        return_attention=True,
    )
    assert output[0].hypotheses[0] == ["<unk>"]
    assert output[0].scores[0] == 0
    assert output[0].attention[0] == [[0]]

    output = translator.score_batch([tokens], [["</s>", "a", "t", "z", "m", "o", "n"]])

    expected_log_probs = [
        -0.036438941955566406,
        -0.20800018310546875,
        -0.06881046295166016,
        -0.5400338172912598,
        -0.12033462524414062,
        -0.1102609634399414,
        -0.044970035552978516,
    ]

    assert output[0].tokens == ["a", "t", "z", "m", "o", "n", "</s>"]
    assert output[0].log_probs == pytest.approx(expected_log_probs, 1e-5)

    # In this mode, an empty target is not enough tokens to run the score so the output is empty.
    output = translator.score_batch([tokens], [[]])
    assert not output[0].tokens
    assert not output[0].log_probs
