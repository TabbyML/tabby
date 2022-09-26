import copy
import inspect
import os
import shutil
import sys

import numpy as np
import pytest

import ctranslate2

from ctranslate2.converters import opennmt_tf
from ctranslate2.converters import utils as conversion_utils
from ctranslate2.specs import common_spec, transformer_spec
from ctranslate2.specs.model_spec import OPTIONAL, index_spec

_TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "tests", "data"
)


def _get_model_path():
    return os.path.join(_TEST_DATA_DIR, "models", "v2", "aren-transliteration")


def _get_model_path_with_vmap(tmpdir, tokens):
    model_dir = str(tmpdir.join("model"))
    shutil.copytree(_get_model_path(), model_dir)
    with open(os.path.join(model_dir, "vmap.txt"), "w", encoding="utf-8") as vmap:
        vmap.write("\t%s\n" % " ".join(tokens))
    return model_dir


def _get_transliterator():
    return ctranslate2.Translator(_get_model_path())


def _write_tokens(batch_tokens, path):
    with open(path, "w", encoding="utf-8") as f:
        for tokens in batch_tokens:
            f.write(" ".join(tokens))
            f.write("\n")


def test_invalid_model_path():
    with pytest.raises(RuntimeError, match="open file"):
        ctranslate2.Translator("xxx")


def test_invalid_model_type():
    with pytest.raises(RuntimeError, match="cannot be used"):
        ctranslate2.Generator(_get_model_path())


def test_invalid_device_settings():
    model_path = _get_model_path()
    with pytest.raises(ValueError, match="device index"):
        ctranslate2.Translator(model_path, device_index=[])
    with pytest.raises(ValueError, match="device index"):
        ctranslate2.Translator(model_path, device_index=[0, 1])


def test_contains_model(tmpdir):
    assert ctranslate2.contains_model(_get_model_path())

    model_dir = tmpdir.join("model")
    model_dir.ensure(dir=1)
    assert not ctranslate2.contains_model(str(model_dir))
    model_dir.join("model.bin").ensure(file=1)
    assert ctranslate2.contains_model(str(model_dir))


def test_get_supported_compute_types():
    assert ctranslate2.get_supported_compute_types("cpu") == {"float", "int16", "int8"}


def test_translator_properties():
    translator = ctranslate2.Translator(_get_model_path(), inter_threads=2)
    assert translator.model_is_loaded
    assert translator.device == "cpu"
    assert translator.device_index == [0, 0]
    assert translator.num_translators == 2
    assert translator.num_queued_batches == 0


def test_compute_type():
    model_path = _get_model_path()
    with pytest.raises(ValueError, match="compute type"):
        ctranslate2.Translator(model_path, compute_type="float64")
    with pytest.raises(TypeError, match="incompatible constructor arguments"):
        ctranslate2.Translator(model_path, compute_type=["int8", "int16"])
    ctranslate2.Translator(model_path, compute_type="int8")
    ctranslate2.Translator(model_path, compute_type={"cuda": "float16", "cpu": "int8"})


@pytest.mark.parametrize("max_batch_size", [0, 1])
def test_batch_translation(max_batch_size):
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"], ["آ", "ت", "ش", "ي", "س", "و", "ن"]],
        max_batch_size=max_batch_size,
        return_scores=True,
    )
    assert len(output) == 2
    assert output[0].hypotheses == [["a", "t", "z", "m", "o", "n"]]
    assert output[1].hypotheses == [["a", "c", "h", "i", "s", "o", "n"]]
    assert output[0].scores[0] < 0
    assert not output[0].attention

    expected_repr = "TranslationResult(hypotheses=%s, scores=%s, attention=[])" % (
        output[0].hypotheses,
        output[0].scores,
    )
    assert repr(output[0]) == expected_repr

    # Check backward compatibility with previous result format.
    assert len(output[0]) == 1  # One hypothesis.
    assert len(output[1]) == 1
    assert output[0][0]["tokens"] == ["a", "t", "z", "m", "o", "n"]
    assert output[0][0]["score"] < 0
    assert "attention" not in output[0][0]
    assert output[1][0]["tokens"] == ["a", "c", "h", "i", "s", "o", "n"]


def test_batch_translation_async():
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"], ["آ", "ت", "ش", "ي", "س", "و", "ن"]],
        asynchronous=True,
    )
    assert translator.num_active_batches == 1
    assert output[0].result().hypotheses == [["a", "t", "z", "m", "o", "n"]]
    assert output[1].result().hypotheses == [["a", "c", "h", "i", "s", "o", "n"]]
    assert output[0].done()
    assert output[1].done()


def test_iterable_translation():
    source = [["آ", "ت", "ز", "م", "و", "ن"], ["آ", "ت", "ش", "ي", "س", "و", "ن"]]
    translator = _get_transliterator()
    results = translator.translate_iterable(iter(source), return_scores=True)
    assert inspect.isgenerator(results)
    results = list(results)
    assert results[0].hypotheses == [["a", "t", "z", "m", "o", "n"]]
    assert results[0].scores
    assert results[1].hypotheses == [["a", "c", "h", "i", "s", "o", "n"]]
    assert results[1].scores

    target_prefix = [["a", "t", "s"], ["a", "c", "h", "e"]]
    results = translator.translate_iterable(iter(source), iter(target_prefix))
    results = list(results)
    assert results[0].hypotheses == [["a", "t", "s", "u", "m", "o", "n"]]
    assert not results[0].scores
    assert results[1].hypotheses == [["a", "c", "h", "e", "s", "o", "n"]]
    assert not results[1].scores

    target_prefix.pop()
    with pytest.raises(ValueError, match="length"):
        next(translator.translate_iterable(iter(source), iter(target_prefix)))

    with pytest.raises(StopIteration):
        next(translator.translate_iterable(iter([])))


def test_file_translation(tmpdir):
    input_path = str(tmpdir.join("input.txt"))
    output_path = str(tmpdir.join("output.txt"))
    with open(input_path, "w", encoding="utf-8") as input_file:
        input_file.write("آ ت ز م و ن")
        input_file.write("\n")
        input_file.write("آ ت ش ي س و ن")
        input_file.write("\n")
    translator = _get_transliterator()
    stats = translator.translate_file(input_path, output_path)
    with open(output_path, encoding="utf-8") as output_file:
        lines = output_file.readlines()
        assert lines[0].strip() == "a t z m o n"
        assert lines[1].strip() == "a c h i s o n"
    assert stats.num_tokens == 13
    assert stats.num_examples == 2
    assert isinstance(stats.total_time_in_ms, float)

    expected_repr = (
        "TranslationStats(num_tokens=13, num_examples=2, total_time_in_ms=%s)"
        % stats.total_time_in_ms
    )
    assert repr(stats) == expected_repr

    # For backward compatibility, ensure that stats is tuple-like.
    num_tokens, num_examples, total_time_in_ms = stats
    assert stats[0] == num_tokens
    assert stats[1] == num_examples
    assert stats[2] == total_time_in_ms


def test_raw_file_translation(tmpdir):
    input_path = str(tmpdir.join("input.txt"))
    output_path = str(tmpdir.join("output.txt"))
    with open(input_path, "w", encoding="utf-8") as input_file:
        input_file.write("آتزمون")
        input_file.write("\n")
        input_file.write("آتشيسون")
        input_file.write("\n")

    translator = ctranslate2.Translator(_get_model_path())
    tokenize_fn = lambda text: list(text)
    detokenize_fn = lambda tokens: "".join(tokens)

    with pytest.raises(ValueError, match="target_detokenize_fn"):
        translator.translate_file(
            input_path, output_path, source_tokenize_fn=tokenize_fn
        )
    with pytest.raises(ValueError, match="source_tokenize_fn"):
        translator.translate_file(
            input_path, output_path, target_detokenize_fn=detokenize_fn
        )

    translator.translate_file(
        input_path,
        output_path,
        source_tokenize_fn=tokenize_fn,
        target_detokenize_fn=detokenize_fn,
    )

    with open(output_path, encoding="utf-8") as output_file:
        lines = output_file.readlines()
        assert lines[0].strip() == "atzmon"
        assert lines[1].strip() == "achison"


def test_file_translation_with_prefix(tmpdir):
    source_path = str(tmpdir.join("input.txt"))
    target_path = str(tmpdir.join("target.txt"))
    output_path = str(tmpdir.join("output.txt"))
    with open(source_path, "w", encoding="utf-8") as source_file:
        source_file.write("آ ت ز م و ن")
        source_file.write("\n")
        source_file.write("آ ت ش ي س و ن")
        source_file.write("\n")
    with open(target_path, "w", encoding="utf-8") as target_file:
        target_file.write("a t s\n")

    translator = _get_transliterator()

    with pytest.raises(RuntimeError, match="has less examples"):
        # One line is missing from target_path.
        translator.translate_file(
            source_path,
            output_path,
            target_path=target_path,
        )

    with open(target_path, "a", encoding="utf-8") as target_file:
        target_file.write("\n")  # No prefix.

    translator.translate_file(
        source_path,
        output_path,
        target_path=target_path,
    )

    with open(output_path, encoding="utf-8") as output_file:
        lines = output_file.readlines()
        assert lines[0].strip() == "a t s u m o n"
        assert lines[1].strip() == "a c h i s o n"


def test_raw_file_translation_with_prefix(tmpdir):
    source_path = str(tmpdir.join("input.txt"))
    target_path = str(tmpdir.join("target.txt"))
    output_path = str(tmpdir.join("output.txt"))
    with open(source_path, "w", encoding="utf-8") as source_file:
        source_file.write("آتزمون")
        source_file.write("\n")
        source_file.write("آتشيسون")
        source_file.write("\n")
    with open(target_path, "w", encoding="utf-8") as target_file:
        # Write target in reverse to use a different tokenization.
        target_file.write("sta\n")
        target_file.write("\n")

    translator = ctranslate2.Translator(_get_model_path())
    source_tokenize_fn = lambda text: list(text)
    target_tokenize_fn = lambda text: list(reversed(list(text)))
    detokenize_fn = lambda tokens: "".join(tokens)

    with pytest.raises(ValueError, match="target_tokenize_fn"):
        # Target tokenization is missing.
        translator.translate_file(
            source_path,
            output_path,
            target_path=target_path,
            source_tokenize_fn=source_tokenize_fn,
            target_detokenize_fn=detokenize_fn,
        )

    translator.translate_file(
        source_path,
        output_path,
        target_path=target_path,
        source_tokenize_fn=source_tokenize_fn,
        target_tokenize_fn=target_tokenize_fn,
        target_detokenize_fn=detokenize_fn,
    )

    with open(output_path, encoding="utf-8") as output_file:
        lines = output_file.readlines()
        assert lines[0].strip() == "atsumon"
        assert lines[1].strip() == "achison"


def test_empty_translation():
    translator = _get_transliterator()
    assert translator.translate_batch([]) == []


def test_invalid_translation_options():
    translator = _get_transliterator()
    with pytest.raises(ValueError, match="is greater than"):
        translator.translate_batch(
            [["آ", "ت", "ز", "م", "و", "ن"]],
            min_decoding_length=10,
            max_decoding_length=5,
        )


def test_invalid_translation_options_async():
    translator = _get_transliterator()
    outputs = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"]],
        min_decoding_length=10,
        max_decoding_length=5,
        asynchronous=True,
    )

    # All calls to result() should raise the exception.
    for _ in range(2):
        with pytest.raises(ValueError, match="is greater than"):
            outputs[0].result()
        assert outputs[0].done()


def test_hard_target_prefix():
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"], ["آ", "ت", "ش", "ي", "س", "و", "ن"]],
        target_prefix=[["a", "t", "s"], None],
    )
    assert output[0].hypotheses[0][:3] == ["a", "t", "s"]
    assert output[1].hypotheses[0] == ["a", "c", "h", "i", "s", "o", "n"]


@pytest.mark.parametrize("beam_size", [1, 2])
def test_hard_target_prefix_with_vmap(tmpdir, beam_size):
    model_dir = _get_model_path_with_vmap(tmpdir, ["t", "z", "m", "o", "n"])
    translator = ctranslate2.Translator(model_dir)
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"]],
        target_prefix=[["a", "t", "z"]],
        beam_size=beam_size,
        use_vmap=True,
    )
    assert output[0].hypotheses[0] == ["a", "t", "z", "m", "o", "n"]


def test_strongly_biased_target_prefix():
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"], ["آ", "ت", "ش", "ي", "س", "و", "ن"]],
        target_prefix=[["a", "t", "s"], None],
        prefix_bias_beta=0.9999999,
    )
    assert output[0].hypotheses[0][:3] == ["a", "t", "s"]
    assert output[1].hypotheses[0] == ["a", "c", "h", "i", "s", "o", "n"]


def test_weakly_biased_target_prefix():
    translator = _get_transliterator()
    unconstrained_output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"], ["آ", "ت", "ش", "ي", "س", "و", "ن"]],
        return_scores=True,
    )
    weakly_biased_output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"], ["آ", "ت", "ش", "ي", "س", "و", "ن"]],
        target_prefix=[["a", "t", "s"], ["s", "i", "o"]],
        prefix_bias_beta=0.0000001,
        return_scores=True,
    )
    assert (
        unconstrained_output[0].hypotheses[0] == weakly_biased_output[0].hypotheses[0]
    )
    assert (
        abs(unconstrained_output[0].scores[0] - weakly_biased_output[0].scores[0])
        < 0.00001
    )

    assert (
        unconstrained_output[1].hypotheses[0] == weakly_biased_output[1].hypotheses[0]
    )
    assert (
        abs(unconstrained_output[1].scores[0] - weakly_biased_output[1].scores[0])
        < 0.00001
    )


@pytest.mark.parametrize("beam_size", [1, 2])
def test_repetition_penalty_with_vmap(tmpdir, beam_size):
    model_dir = _get_model_path_with_vmap(tmpdir, ["a", "t", "z", "m", "o", "n"])
    translator = ctranslate2.Translator(model_dir)
    output = translator.translate_batch(
        [["ن"] * 3],
        repetition_penalty=100,
        max_decoding_length=3,
        beam_size=beam_size,
        use_vmap=True,
    )
    tokens = output[0].hypotheses[0]
    assert len(tokens) == len(set(tokens))


@pytest.mark.parametrize("beam_size", [1, 2])
def test_no_repeat_ngram_size_with_vmap(tmpdir, beam_size):
    model_dir = _get_model_path_with_vmap(tmpdir, ["a", "t", "z", "m", "o", "n"])
    translator = ctranslate2.Translator(model_dir)
    no_repeat_ngram_size = 3
    output = translator.translate_batch(
        [["ن"] * 50],
        no_repeat_ngram_size=no_repeat_ngram_size,
        beam_size=beam_size,
        use_vmap=True,
    )
    tokens = output[0].hypotheses[0]
    ngrams = [
        "".join(tokens[i : i + no_repeat_ngram_size])
        for i in range(len(tokens) - no_repeat_ngram_size + 1)
    ]
    assert len(ngrams) == len(set(ngrams))


def test_num_hypotheses():
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"]], beam_size=4, num_hypotheses=2
    )
    assert len(output[0].hypotheses) == 2


def test_max_decoding_length():
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"]], max_decoding_length=2
    )
    assert output[0].hypotheses[0] == ["a", "t"]


def test_min_decoding_length():
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"]], min_decoding_length=7
    )
    assert len(output[0].hypotheses[0]) > 6  # 6 is the expected target length.


@pytest.mark.parametrize("beam_size", [1, 2])
def test_min_decoding_length_with_vmap(tmpdir, beam_size):
    model_dir = _get_model_path_with_vmap(tmpdir, ["a", "t", "z", "m", "o", "n"])
    translator = ctranslate2.Translator(model_dir)
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"]],
        min_decoding_length=1,
        beam_size=beam_size,
        use_vmap=True,
    )
    assert output[0].hypotheses[0] == ["a", "t", "z", "m", "o", "n"]


def test_return_attention():
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"]], return_attention=True
    )
    attention = output[0].attention[0]
    assert len(attention) == 6  # Target length.
    for vector in attention:
        assert len(vector) == 6  # Source length.
        assert all(isinstance(value, float) for value in vector)


def test_ignore_scores():
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"]], beam_size=1, return_scores=False
    )
    assert not output[0].scores


def test_return_alternatives():
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"]],
        target_prefix=[["a", "t"]],
        num_hypotheses=10,
        return_alternatives=True,
    )
    assert len(output[0].hypotheses) == 10
    assert output[0].hypotheses[0] == ["a", "t", "z", "m", "o", "n"]
    assert output[0].hypotheses[1] == ["a", "t", "s", "u", "m", "o", "n"]


def test_return_alternatives_with_vmap(tmpdir):
    model_dir = _get_model_path_with_vmap(tmpdir, ["z", "s", "u", "m", "o", "n"])
    translator = ctranslate2.Translator(model_dir)
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"]],
        target_prefix=[["a", "t"]],
        num_hypotheses=2,
        return_alternatives=True,
        use_vmap=True,
    )
    assert len(output[0].hypotheses) == 2
    assert output[0].hypotheses[0] == ["a", "t", "z", "m", "o", "n"]
    assert output[0].hypotheses[1] == ["a", "t", "s", "u", "m", "o", "n"]


def test_random_sampling():
    ctranslate2.set_random_seed(46)
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"]], beam_size=1, sampling_topk=0
    )
    assert output[0].hypotheses[0] == ["a", "t", "z", "m", "u", "n"]


def test_score_api(tmpdir):
    source = [
        ["آ", "ت", "ز", "م", "و", "ن"],
        ["آ", "ت", "ش", "ي", "س", "و", "ن"],
    ]
    target = [
        ["a", "t", "z", "m", "o", "n"],
        ["a", "c", "h", "i", "s", "o", "n"],
    ]
    expected = [
        [-0.106023, -0.065410, -0.056002, -0.447953, -0.230714, -0.092184, -0.063463],
        [
            -0.072660,
            -0.300309,
            -0.181187,
            -0.395671,
            -0.025631,
            -0.123466,
            -0.002034,
            -0.012639,
        ],
    ]

    translator = _get_transliterator()

    # Check the different ways of getting the log probs.
    all_log_probs = [
        # Backward compatibility with reading the result as a list of log probs.
        translator.score_batch(source, target),
        [result.log_probs for result in translator.score_batch(source, target)],
        [
            async_result.result().log_probs
            for async_result in translator.score_batch(
                source, target, asynchronous=True
            )
        ],
        [
            result.log_probs
            for result in translator.score_iterable(iter(source), iter(target))
        ],
    ]

    for batch_log_probs in all_log_probs:
        for log_probs, expected_log_probs in zip(batch_log_probs, expected):
            np.testing.assert_allclose(log_probs, expected_log_probs, rtol=1e-4)

    source_path = str(tmpdir.join("source.txt"))
    target_path = str(tmpdir.join("target.txt"))
    output_path = str(tmpdir.join("output.txt"))
    _write_tokens(source, source_path)
    _write_tokens(target, target_path)

    stats = translator.score_file(
        source_path,
        target_path,
        output_path,
        with_tokens_score=True,
    )
    assert stats.num_examples == 2
    assert stats.num_tokens == 15

    with open(output_path, encoding="utf-8") as output_file:
        for line, expected_tokens, expected_scores in zip(
            output_file, target, expected
        ):
            parts = line.strip().split("|||")
            assert len(parts) == 3

            mean_score = float(parts[0].strip())
            tokens = parts[1].split()
            scores = list(map(float, parts[2].split()))

            assert tokens == expected_tokens + ["</s>"]
            assert mean_score == pytest.approx(np.mean(expected_scores), 1e-4)
            assert scores == pytest.approx(expected_scores, 1e-4)

    # Test empty inputs.
    assert translator.score_batch([], []) == []

    with pytest.raises(ValueError, match="length"):
        next(translator.score_iterable(iter([["a"]]), iter([])))
    with pytest.raises(StopIteration):
        next(translator.score_iterable(iter([]), iter([])))

    output = translator.score_batch([[]], [[]])
    assert output[0].tokens == ["</s>"]
    assert output[0].log_probs == [0]

    output = translator.score_batch([[], []], target)
    assert output[0].tokens == ["a", "t", "z", "m", "o", "n", "</s>"]
    assert output[0].log_probs == [0, 0, 0, 0, 0, 0, 0]
    assert output[1].tokens == ["a", "c", "h", "i", "s", "o", "n", "</s>"]
    assert output[1].log_probs == [0, 0, 0, 0, 0, 0, 0, 0]


@pytest.mark.parametrize("to_cpu", [False, True])
def test_model_unload(to_cpu):
    batch = [["آ", "ت", "ز", "م", "و", "ن"]]
    translator = _get_transliterator()
    translator.unload_model(to_cpu=to_cpu)
    if not to_cpu:
        assert not translator.model_is_loaded
        with pytest.raises(RuntimeError, match="unloaded"):
            translator.translate_batch(batch)
    else:
        assert translator.model_is_loaded
    translator.load_model()
    assert translator.model_is_loaded
    output = translator.translate_batch(batch)
    assert len(output) == 1
    assert output[0].hypotheses[0] == ["a", "t", "z", "m", "o", "n"]


def test_model_unload_while_async_translation():
    translator = _get_transliterator()
    outputs = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"]], asynchronous=True
    )
    translator.unload_model()
    assert translator.model_is_loaded
    assert outputs[0].result().hypotheses[0] == ["a", "t", "z", "m", "o", "n"]


skip_if_data_missing = pytest.mark.skipif(
    not os.path.isdir(
        os.path.join(_TEST_DATA_DIR, "models", "transliteration-aren-all")
    ),
    reason="Data files are not available",
)
skip_on_windows = pytest.mark.skipif(
    sys.platform == "win32", reason="Test case disabled on Windows"
)
only_on_linux = pytest.mark.skipif(
    sys.platform != "linux", reason="Test case only running on Linux"
)


@skip_if_data_missing
@pytest.mark.parametrize(
    "model_path,src_vocab,tgt_vocab,model_spec",
    [
        (
            "v1/checkpoint",
            "ar.vocab",
            "en.vocab",
            ctranslate2.specs.TransformerSpec(6, 8),
        ),
        (
            "v1/checkpoint",
            "ar.vocab",
            "en.vocab",
            None,
        ),
        (
            "v2/checkpoint",
            "ar.vocab",
            "en.vocab",
            ctranslate2.specs.TransformerSpec(6, 8),
        ),
        (
            "v2/checkpoint",
            "ar.vocab",
            "en.vocab",
            None,
        ),
    ],
)
def test_opennmt_tf_model_conversion(
    tmpdir, model_path, src_vocab, tgt_vocab, model_spec
):
    model_path = os.path.join(
        _TEST_DATA_DIR, "models", "transliteration-aren-all", "opennmt_tf", model_path
    )
    src_vocab = os.path.join(model_path, src_vocab)
    tgt_vocab = os.path.join(model_path, tgt_vocab)

    if model_spec is None:
        config = {
            "model_dir": model_path,
            "data": {"source_vocabulary": src_vocab, "target_vocabulary": tgt_vocab},
        }
        original_config = copy.deepcopy(config)
        converter = ctranslate2.converters.OpenNMTTFConverterV2.from_config(
            config, auto_config=True
        )

        # auto_config should not update the configuration in place.
        assert config == original_config
    else:
        converter = ctranslate2.converters.OpenNMTTFConverter(
            model_spec,
            src_vocab,
            tgt_vocab,
            model_path=model_path,
        )
    output_dir = str(tmpdir.join("ctranslate2_model"))
    converter.convert(output_dir)

    src_vocab_path = os.path.join(output_dir, "source_vocabulary.txt")
    tgt_vocab_path = os.path.join(output_dir, "target_vocabulary.txt")

    # Check lines end with \n on all platforms.
    with open(src_vocab_path, encoding="utf-8", newline="") as vocab_file:
        assert vocab_file.readline() == "<blank>\n"
    with open(tgt_vocab_path, encoding="utf-8", newline="") as vocab_file:
        assert vocab_file.readline() == "<blank>\n"

    translator = ctranslate2.Translator(output_dir)
    output = translator.translate_batch([["آ", "ت", "ز", "م", "و", "ن"]])
    assert output[0].hypotheses[0] == ["a", "t", "z", "m", "o", "n"]


@skip_if_data_missing
@pytest.mark.parametrize("quantization", ["float16", "int16", "int8", "int8_float16"])
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
        ctranslate2.specs.TransformerSpec(6, 8),
        os.path.join(model_path, "ar.vocab"),
        os.path.join(model_path, "en.vocab"),
        model_path=model_path,
    )
    output_dir = str(tmpdir.join("ctranslate2_model"))
    converter.convert(output_dir, quantization=quantization)
    translator = ctranslate2.Translator(output_dir)
    output = translator.translate_batch([["آ", "ت", "ز", "م", "و", "ن"]])
    assert output[0].hypotheses[0] == ["a", "t", "z", "m", "o", "n"]


@skip_if_data_missing
def test_opennmt_tf_variables_conversion(tmpdir):
    import opennmt

    model_path = os.path.join(
        _TEST_DATA_DIR,
        "models",
        "transliteration-aren-all",
        "opennmt_tf",
        "v2",
        "checkpoint",
    )

    src_vocab = opennmt.data.Vocab.from_file(os.path.join(model_path, "ar.vocab"))
    tgt_vocab = opennmt.data.Vocab.from_file(os.path.join(model_path, "en.vocab"))
    _, variables = opennmt_tf.load_model(model_path)
    converter = ctranslate2.converters.OpenNMTTFConverter(
        ctranslate2.specs.TransformerSpec(6, 8),
        src_vocab,
        tgt_vocab,
        variables=variables,
    )
    output_dir = str(tmpdir.join("ctranslate2_model"))
    converter.convert(output_dir)
    translator = ctranslate2.Translator(output_dir)
    output = translator.translate_batch([["آ", "ت", "ز", "م", "و", "ن"]])
    assert output[0].hypotheses[0] == ["a", "t", "z", "m", "o", "n"]


@skip_if_data_missing
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
        ctranslate2.specs.TransformerSpec(6, 8),
        os.path.join(model_path, "en.vocab"),
        os.path.join(model_path, "ar.vocab"),
        model_path=model_path,
    )
    output_dir = str(tmpdir.join("ctranslate2_model"))
    with pytest.raises(ValueError, match="expected a vocabulary of size"):
        converter.convert(output_dir)


def _create_vocab(tmpdir, name="vocab", size=10):
    import opennmt

    vocab = opennmt.data.Vocab()
    for i in range(size):
        vocab.add(str(i))
    vocab_path = str(tmpdir.join("%s.txt" % name))
    vocab.serialize(vocab_path)
    return vocab_path


def _create_checkpoint(model, tmpdir):
    import opennmt
    import tensorflow as tf

    vocab_path = _create_vocab(tmpdir)
    model.initialize({"source_vocabulary": vocab_path, "target_vocabulary": vocab_path})
    model.create_variables()

    checkpoint_prefix = str(tmpdir.join("ckpt"))
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_path = checkpoint.write(checkpoint_prefix)
    return checkpoint_path, vocab_path


def test_opennmt_tf_model_conversion_invalid_dir(tmpdir):
    model_path = str(tmpdir.join("model").ensure(dir=1))
    vocab_path = _create_vocab(tmpdir)
    converter = ctranslate2.converters.OpenNMTTFConverter(
        ctranslate2.specs.TransformerSpec(6, 8),
        vocab_path,
        vocab_path,
        model_path=model_path,
    )
    output_dir = str(tmpdir.join("ctranslate2_model"))
    with pytest.raises(ValueError, match="not found"):
        converter.convert(output_dir)


def test_opennmt_tf_shared_embeddings_conversion(tmpdir):
    # Issue https://github.com/OpenNMT/CTranslate2/issues/118
    import opennmt

    model = opennmt.models.Transformer(
        opennmt.inputters.WordEmbedder(32),
        opennmt.inputters.WordEmbedder(32),
        num_layers=3,
        num_units=32,
        num_heads=4,
        ffn_inner_dim=64,
        share_embeddings=opennmt.models.EmbeddingsSharingLevel.ALL,
    )

    model_path, vocab_path = _create_checkpoint(model, tmpdir)

    converter = ctranslate2.converters.OpenNMTTFConverter(
        model.ctranslate2_spec,
        vocab_path,
        vocab_path,
        model_path=model_path,
    )
    output_dir = str(tmpdir.join("ctranslate2_model"))
    converter.convert(output_dir)

    assert os.path.isfile(os.path.join(output_dir, "shared_vocabulary.txt"))

    # Check that the translation runs.
    translator = ctranslate2.Translator(output_dir)
    translator.translate_batch([["1", "2", "3"]], max_decoding_length=10)


def test_opennmt_tf_postnorm_transformer_conversion(tmpdir):
    import opennmt

    model = opennmt.models.Transformer(
        opennmt.inputters.WordEmbedder(32),
        opennmt.inputters.WordEmbedder(32),
        num_layers=3,
        num_units=32,
        num_heads=4,
        ffn_inner_dim=64,
        pre_norm=False,
    )

    model_path, vocab_path = _create_checkpoint(model, tmpdir)
    converter = ctranslate2.converters.OpenNMTTFConverter(
        model.ctranslate2_spec,
        vocab_path,
        vocab_path,
        model_path=model_path,
    )
    output_dir = str(tmpdir.join("ctranslate2_model"))
    converter.convert(output_dir)


def test_opennmt_tf_gpt_conversion(tmpdir):
    import opennmt

    vocabulary = _create_vocab(tmpdir, "vocab")
    model = opennmt.models.GPT2Small()
    model.initialize(dict(vocabulary=vocabulary))
    model.create_variables()

    output_dir = str(tmpdir.join("ctranslate2_model"))
    converter = ctranslate2.converters.OpenNMTTFConverterV2(model)
    converter.convert(output_dir)

    assert os.path.isfile(os.path.join(output_dir, "vocabulary.txt"))


def test_opennmt_tf_multi_features(tmpdir):
    import opennmt

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
            "source_1_vocabulary": _create_vocab(tmpdir, "source_1", 50),
            "source_2_vocabulary": _create_vocab(tmpdir, "source_2", 10),
            "target_vocabulary": _create_vocab(tmpdir, "target", 60),
        }
    )
    model.create_variables()

    converter = ctranslate2.converters.OpenNMTTFConverterV2(model)
    output_dir = str(tmpdir.join("ctranslate2_model"))
    converter.convert(output_dir)

    assert os.path.isfile(os.path.join(output_dir, "source_1_vocabulary.txt"))
    assert os.path.isfile(os.path.join(output_dir, "source_2_vocabulary.txt"))


@skip_if_data_missing
@skip_on_windows
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
    converter.convert(output_dir)
    translator = ctranslate2.Translator(output_dir)
    output = translator.translate_batch([["آ", "ت", "ز", "م", "و", "ن"]])
    assert output[0].hypotheses[0] == ["a", "t", "z", "m", "o", "n"]


@skip_if_data_missing
@skip_on_windows
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
    converter.convert(output_dir)
    translator = ctranslate2.Translator(output_dir)
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"], ["آ", "ر", "ث", "ر"]]
    )
    assert output[0].hypotheses[0] == ["a", "t", "z", "o", "m", "o", "n"]
    assert output[1].hypotheses[0] == ["a", "r", "t", "h", "e", "r"]


@skip_if_data_missing
@skip_on_windows
@pytest.mark.parametrize(
    "filename", ["aren_features_concat_10000.pt", "aren_features_sum_10000.pt"]
)
def test_opennmt_py_source_features(tmpdir, filename):
    model_path = os.path.join(
        _TEST_DATA_DIR,
        "models",
        "transliteration-aren-all",
        "opennmt_py",
        filename,
    )
    converter = ctranslate2.converters.OpenNMTPyConverter(model_path)
    output_dir = str(tmpdir.join("ctranslate2_model"))
    converter.convert(output_dir)
    assert os.path.isfile(os.path.join(output_dir, "source_1_vocabulary.txt"))
    assert os.path.isfile(os.path.join(output_dir, "source_2_vocabulary.txt"))

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

    input_path = str(tmpdir.join("input.txt"))
    output_path = str(tmpdir.join("output.txt"))

    _write_tokens(source, input_path)
    with pytest.raises(ValueError, match="features"):
        translator.translate_file(input_path, output_path)

    _write_tokens(source_w_features, input_path)
    translator.translate_file(input_path, output_path)
    with open(output_path) as output_file:
        for line, expected_hypothesis in zip(output_file, expected_target):
            assert line.strip().split() == expected_hypothesis


@skip_on_windows
def test_opennmt_py_transformer_lm(tmpdir):
    model_path = os.path.join(_TEST_DATA_DIR, "models", "pi_lm_step_5000.pt")
    if not os.path.exists(model_path):
        pytest.skip("Checkpoint file is not available")

    converter = ctranslate2.converters.OpenNMTPyConverter(model_path)
    output_dir = str(tmpdir.join("ctranslate2_model"))
    converter.convert(output_dir)

    generator = ctranslate2.Generator(output_dir)
    results = generator.generate_batch([["<s>", "3", ".", "1", "4"]], max_length=12)

    assert "".join(results[0].sequences[0]) == "3.1415926535"


@skip_if_data_missing
@skip_on_windows
def test_fairseq_model_conversion(tmpdir):
    data_dir = os.path.join(
        _TEST_DATA_DIR,
        "models",
        "transliteration-aren-all",
        "fairseq",
    )
    converter = ctranslate2.converters.FairseqConverter(
        os.path.join(data_dir, "model.pt"), data_dir
    )
    output_dir = str(tmpdir.join("ctranslate2_model"))
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


@skip_if_data_missing
@skip_on_windows
def test_fairseq_user_start_token(tmpdir):
    data_dir = os.path.join(
        _TEST_DATA_DIR,
        "models",
        "transliteration-aren-all",
        "fairseq",
    )
    converter = ctranslate2.converters.FairseqConverter(
        os.path.join(data_dir, "model.pt"), data_dir, no_default_special_tokens=True
    )
    output_dir = str(tmpdir.join("ctranslate2_model"))
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


@skip_if_data_missing
def test_marian_model_conversion(tmpdir):
    model_dir = os.path.join(_TEST_DATA_DIR, "models", "opus-mt-ende")
    converter = ctranslate2.converters.OpusMTConverter(model_dir)
    output_dir = str(tmpdir.join("ctranslate2_model"))
    converter.convert(output_dir)
    translator = ctranslate2.Translator(output_dir)
    output = translator.translate_batch([["▁Hello", "▁world", "!"]])
    assert output[0].hypotheses[0] == ["▁Hallo", "▁Welt", "!"]


@pytest.fixture
def clear_transformers_cache():
    """Clears the Transformers model cache after each test when running in a CI."""
    import transformers

    yield

    if os.environ.get("CI") == "true":
        shutil.rmtree(transformers.utils.default_cache_path)


_TRANSFORMERS_TRANSLATION_TESTS = [
    (
        "Helsinki-NLP/opus-mt-en-de",
        "▁Hello ▁world ! </s>",
        "",
        "▁Hallo ▁Welt !",
    ),
    (
        "Helsinki-NLP/opus-mt-en-roa",
        ">>ind<< ▁The ▁Prime ▁Minister ▁is ▁coming ▁back ▁tomorrow . </s>",
        "",
        "▁Per da na ▁Men teri ▁akan ▁kembali ▁besok .",
    ),
    (
        "Helsinki-NLP/opus-mt-mul-en",
        "▁Bon jo ur ▁le ▁mo nde </s>",
        "",
        "▁Welcome ▁to ▁the ▁World",
    ),
    (
        "facebook/m2m100_418M",
        "__en__ ▁Hello ▁world ! </s>",
        "__de__",
        "__de__ ▁Hallo ▁der ▁Welt !",
    ),
    (
        "facebook/mbart-large-50-many-to-many-mmt",
        "en_XX ▁Hello ▁world ! </s>",
        "de_DE",
        "de_DE ▁Hallo ▁Welt !",
    ),
    (
        "facebook/mbart-large-en-ro",
        "▁UN ▁Chief ▁Say s ▁There ▁Is ▁No ▁Militar y ▁Solution ▁in ▁Syria </s> en_XX",
        "ro_RO",
        "▁Şe ful ▁ONU ▁de cla ră ▁că ▁nu ▁există ▁o ▁solu ţie ▁militar ă ▁în ▁Siria",
    ),
    (
        "facebook/bart-base",
        "<s> UN ĠChief ĠSays ĠThere ĠIs ĠNo <mask> Ġin ĠSyria </s>",
        "",
        "<s> UN ĠChief ĠSays ĠThere ĠIs ĠNo ĠWar Ġin ĠSyria",
    ),
    (
        "google/pegasus-xsum",
        "▁PG & E ▁stated ▁it ▁scheduled ▁the ▁blackout s ▁in ▁response ▁to ▁forecasts "
        "▁for ▁high ▁winds ▁amid ▁dry ▁conditions . ▁The ▁aim ▁is ▁to ▁reduce ▁the "
        "▁risk ▁of ▁wildfires . ▁Nearly ▁800 ▁thousand ▁customers ▁were ▁scheduled ▁to "
        "▁be ▁affected ▁by ▁the ▁shutoff s ▁which ▁were ▁expected ▁to ▁last ▁through "
        "▁at ▁least ▁midday ▁tomorrow . </s>",
        "",
        "▁California ' s ▁largest ▁electricity ▁provider ▁has ▁turned ▁off ▁power ▁to "
        "▁hundreds ▁of ▁thousands ▁of ▁customers .",
    ),
    (
        "facebook/nllb-200-distilled-600M",
        ["▁Hello ▁world ! </s> eng_Latn", "</s> eng_Latn"],
        ["fra_Latn", "fra_Latn"],
        ["fra_Latn ▁Bon jour ▁le ▁monde ▁!", "fra_Latn"],
    ),
]


@only_on_linux
@pytest.mark.parametrize(
    "model,source_tokens,target_tokens,expected_tokens",
    _TRANSFORMERS_TRANSLATION_TESTS,
    ids=[args[0] for args in _TRANSFORMERS_TRANSLATION_TESTS],
)
def test_transformers_translation(
    clear_transformers_cache,
    tmpdir,
    model,
    source_tokens,
    target_tokens,
    expected_tokens,
):
    converter = ctranslate2.converters.TransformersConverter(model)
    output_dir = str(tmpdir.join("ctranslate2_model"))
    output_dir = converter.convert(output_dir)

    if not isinstance(expected_tokens, list):
        expected_tokens = [expected_tokens]
    if not isinstance(source_tokens, list):
        source_tokens = [source_tokens]
    if target_tokens and not isinstance(target_tokens, list):
        target_tokens = [target_tokens]

    translator = ctranslate2.Translator(output_dir)
    results = translator.translate_batch(
        [line.split() for line in source_tokens],
        [line.split() for line in target_tokens] if target_tokens else None,
    )
    output_tokens = [" ".join(result.hypotheses[0]) for result in results]
    assert output_tokens == expected_tokens


_TRANSFORMERS_GENERATION_TESTS = [
    (
        "gpt2",
        "<|endoftext|>",
        10,
        "Ċ The Ġfirst Ġtime ĠI Ġsaw Ġthe Ġnew Ġversion Ġof",
    ),
    (
        "facebook/opt-350m",
        "</s>",
        10,
        "Ċ The Ġfollowing Ġis Ġa Ġlist Ġof Ġthe Ġmost Ġpopular",
    ),
    (
        "microsoft/DialoGPT-medium",
        "Hello <|endoftext|>",
        100,
        "Hello <|endoftext|> Hello Ġ! Ġ: D",
    ),
]


@only_on_linux
@pytest.mark.parametrize(
    "model,start_tokens,max_length,expected_tokens",
    _TRANSFORMERS_GENERATION_TESTS,
    ids=[args[0] for args in _TRANSFORMERS_GENERATION_TESTS],
)
def test_transformers_generation(
    clear_transformers_cache,
    tmpdir,
    model,
    start_tokens,
    max_length,
    expected_tokens,
):
    converter = ctranslate2.converters.TransformersConverter(model)
    output_dir = str(tmpdir.join("ctranslate2_model"))
    output_dir = converter.convert(output_dir)

    generator = ctranslate2.Generator(output_dir)
    results = generator.generate_batch([start_tokens.split()], max_length=max_length)
    output_tokens = " ".join(results[0].sequences[0])
    assert output_tokens == expected_tokens

    # Test empty inputs.
    assert generator.generate_batch([]) == []

    with pytest.raises(ValueError, match="start token"):
        generator.generate_batch([[]])


@only_on_linux
def test_transformers_marianmt_vocabulary(clear_transformers_cache, tmpdir):
    converter = ctranslate2.converters.TransformersConverter(
        "Helsinki-NLP/opus-mt-en-de"
    )
    output_dir = str(tmpdir.join("ctranslate2_model"))
    output_dir = converter.convert(output_dir)

    with open(os.path.join(output_dir, "shared_vocabulary.txt")) as vocab_file:
        vocab = list(line.rstrip("\n") for line in vocab_file)

    assert vocab[-1] != "<pad>"


@only_on_linux
@pytest.mark.parametrize("beam_size", [1, 2])
def test_transformers_marianmt_disable_unk(clear_transformers_cache, tmpdir, beam_size):
    converter = ctranslate2.converters.TransformersConverter(
        "Helsinki-NLP/opus-mt-en-roa"
    )
    output_dir = str(tmpdir.join("ctranslate2_model"))
    output_dir = converter.convert(output_dir)

    tokens = ">>ind<< ▁The ▁Prime <unk> ▁is ▁coming ▁back ▁tomorrow . </s>".split()
    translator = ctranslate2.Translator(output_dir)
    output = translator.translate_batch([tokens], beam_size=beam_size, disable_unk=True)
    assert "<unk>" not in output[0].hypotheses[0]


@only_on_linux
def test_transformers_lm_scoring(tmpdir):
    converter = ctranslate2.converters.TransformersConverter("gpt2")
    output_dir = str(tmpdir.join("ctranslate2_model"))
    output_dir = converter.convert(output_dir)
    generator = ctranslate2.Generator(output_dir)

    tokens = "Ċ The Ġfirst Ġtime ĠI Ġsaw Ġthe Ġnew Ġversion Ġof".split()
    output = generator.score_batch([tokens])[0]
    assert output.tokens == tokens[1:]
    assert len(output.log_probs) == len(output.tokens)

    # Test empty inputs.
    assert generator.score_batch([]) == []

    output = generator.score_batch([[], tokens])[0]
    assert not output.tokens
    assert not output.log_probs

    output = generator.score_batch([["<|endoftext|>"]])[0]
    assert not output.tokens
    assert not output.log_probs


@only_on_linux
def test_transformers_generator_on_iterables(tmpdir):
    converter = ctranslate2.converters.TransformersConverter("gpt2")
    output_dir = str(tmpdir.join("ctranslate2_model"))
    output_dir = converter.convert(output_dir)
    generator = ctranslate2.Generator(output_dir)

    start_tokens = ["<|endoftext|>"]
    tokens = "Ċ The Ġfirst Ġtime ĠI Ġsaw Ġthe Ġnew Ġversion Ġof".split()
    output = next(generator.generate_iterable(iter([start_tokens]), max_length=10))
    assert output.sequences[0] == tokens

    output = next(generator.score_iterable(iter([tokens])))
    assert output.tokens == tokens[1:]
    assert len(output.log_probs) == len(output.tokens)

    # Test empty iterables.
    with pytest.raises(StopIteration):
        next(generator.score_iterable(iter([])))
    with pytest.raises(StopIteration):
        next(generator.generate_iterable(iter([])))


def _array_equal(a, b):
    return a.dtype == b.dtype and np.array_equal(a, b)


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
            self.g = "hello"

    spec = Spec()
    spec.validate()
    assert spec.a.dtype == np.float32
    assert spec.b.dtype == np.float16
    assert spec.c.dtype == np.int32
    assert spec.d == OPTIONAL
    assert spec.e.a.dtype == np.float16
    assert _array_equal(spec.f, np.int8(1))
    assert _array_equal(spec.g, np.array([104, 101, 108, 108, 111], dtype=np.int8))

    with pytest.raises(AttributeError, match="Attribute z does not exist"):
        spec.z = True


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
            self.weight_scale = OPTIONAL
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


def test_int8_quantization():
    class Spec(ctranslate2.specs.LayerSpec):
        def __init__(self):
            self.weight = np.array([[-10, -3, 5, 2], [0, 0, 0, 0]], dtype=np.float32)
            self.weight_scale = OPTIONAL

    spec = Spec()
    spec.optimize(quantization="int8")
    assert _array_equal(
        spec.weight, np.array([[-127, -38, 64, 25], [0, 0, 0, 0]], dtype=np.int8)
    )
    assert _array_equal(spec.weight_scale, np.array([12.7, 1], dtype=np.float32))


@pytest.mark.parametrize(
    "quantization,expected_weight,expected_weight_scale,expected_bias",
    [
        (
            None,
            np.array([[-10, -3, 5, 2]], dtype=np.float32),
            None,
            np.array([4], dtype=np.float32),
        ),
        (
            "float16",
            np.array([[-10, -3, 5, 2]], dtype=np.float16),
            None,
            np.array([4], dtype=np.float16),
        ),
        (
            "int8",
            np.array([[-127, -38, 64, 25]], dtype=np.int8),
            np.array([12.7], dtype=np.float32),
            np.array([4], dtype=np.float32),
        ),
        (
            "int8_float16",
            np.array([[-127, -38, 64, 25]], dtype=np.int8),
            np.array([12.7], dtype=np.float32),
            np.array([4], dtype=np.float16),
        ),
        (
            "int16",
            np.array([[-1024, -307, 512, 205]], dtype=np.int16),
            np.float32(102.4),
            np.array([4], dtype=np.float32),
        ),
    ],
)
def test_fp16_weights(
    quantization, expected_weight, expected_weight_scale, expected_bias
):
    class Spec(ctranslate2.specs.LayerSpec):
        def __init__(self, weight, bias):
            self.weight = weight
            self.weight_scale = OPTIONAL
            self.bias = bias

    weight = np.array([[-10, -3, 5, 2]], dtype=np.float16)
    bias = np.array([4], dtype=np.float16)

    spec = Spec(weight, bias)
    spec.validate()
    spec.optimize(quantization=quantization)

    assert _array_equal(spec.weight, expected_weight)
    assert _array_equal(spec.bias, expected_bias)

    # Check the weights were not copied or converted.
    if quantization == "float16":
        assert spec.weight is weight
        assert spec.bias is bias
    elif quantization == "int8_float16":
        assert spec.bias is bias

    if expected_weight_scale is None:
        assert spec.weight_scale == OPTIONAL
    else:
        assert _array_equal(spec.weight_scale, expected_weight_scale)


def test_index_spec():
    spec = ctranslate2.specs.TransformerSpec(6, 8)
    assert isinstance(
        index_spec(spec, "encoder/layer_5"),
        transformer_spec.TransformerEncoderLayerSpec,
    )
    assert isinstance(
        index_spec(spec, "encoder/layer_5/ffn"), transformer_spec.FeedForwardSpec
    )


def test_fuse_linear_no_bias():
    layers = []
    for _ in range(3):
        spec = common_spec.LinearSpec()
        spec.weight = np.zeros([64, 64], dtype=np.float32)
        layers.append(spec)

    spec = common_spec.LinearSpec()
    conversion_utils.fuse_linear(spec, layers)
    assert spec.weight.shape[0] == 64 * 3
    assert spec.bias == OPTIONAL

    spec = common_spec.LinearSpec()
    layers[1].bias = np.zeros([64])
    with pytest.raises(ValueError, match="Cannot fuse linear layers"):
        conversion_utils.fuse_linear(spec, layers)
