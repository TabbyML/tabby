import inspect
import io
import logging
import os
import shutil

import numpy as np
import pytest
import test_utils

import ctranslate2


def _get_model_path():
    return os.path.join(
        test_utils.get_data_dir(), "models", "v2", "aren-transliteration"
    )


def _get_model_path_with_vmap(tmpdir, tokens):
    model_dir = str(tmpdir.join("model"))
    shutil.copytree(_get_model_path(), model_dir)
    with open(os.path.join(model_dir, "vmap.txt"), "w", encoding="utf-8") as vmap:
        vmap.write("\t%s\n" % " ".join(tokens))
    return model_dir


def _get_transliterator():
    return ctranslate2.Translator(_get_model_path())


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
    compute_types = ctranslate2.get_supported_compute_types("cpu")
    assert "float32" in compute_types
    assert "int8" in compute_types


def test_translator_properties():
    translator = ctranslate2.Translator(_get_model_path(), inter_threads=2)
    assert translator.model_is_loaded
    assert translator.device == "cpu"
    assert translator.device_index == [0]
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
        "ExecutionStats(num_tokens=13, num_examples=2, total_time_in_ms=%s)"
        % stats.total_time_in_ms
    )
    assert repr(stats) == expected_repr


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


@pytest.mark.parametrize("beam_size", [1, 2])
def test_strongly_biased_target_prefix(beam_size):
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"], ["آ", "ت", "ش", "ي", "س", "و", "ن"]],
        target_prefix=[["a", "t", "s"], None],
        beam_size=beam_size,
        prefix_bias_beta=0.9999999,
    )
    assert output[0].hypotheses[0][:3] == ["a", "t", "s"]
    assert output[1].hypotheses[0] == ["a", "c", "h", "i", "s", "o", "n"]


@pytest.mark.parametrize("beam_size", [1, 2])
def test_weakly_biased_target_prefix(beam_size):
    translator = _get_transliterator()
    unconstrained_output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"], ["آ", "ت", "ش", "ي", "س", "و", "ن"]],
        beam_size=beam_size,
        return_scores=True,
    )
    weakly_biased_output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"], ["آ", "ت", "ش", "ي", "س", "و", "ن"]],
        target_prefix=[["a", "t", "s"], ["s", "i", "o"]],
        beam_size=beam_size,
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


@pytest.mark.parametrize("beam_size", [1, 2])
def test_suppress_sequences_with_vmap(tmpdir, beam_size):
    model_dir = _get_model_path_with_vmap(
        tmpdir, ["a", "t", "z", "s", "m", "o", "u", "n"]
    )
    translator = ctranslate2.Translator(model_dir)
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"]],
        suppress_sequences=[["o"], ["t", "z", "m"]],
        beam_size=beam_size,
        use_vmap=True,
    )
    assert output[0].hypotheses[0] == ["a", "t", "z", "u", "m", "u", "n"]


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
        [["آ", "ت", "ز", "م", "و", "ن"]],
        beam_size=1,
        sampling_topk=0,
        num_hypotheses=5,
        return_scores=True,
    )

    assert len(output[0].hypotheses) == 5
    assert output[0].hypotheses[0] == ["a", "t", "z", "u", "m", "u", "n"]
    assert output[0].hypotheses[1] == ["a", "t", "z", "i", "m", "o", "n"]

    assert len(output[0].scores) == 5
    assert output[0].scores == list(sorted(output[0].scores, reverse=True))


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
    test_utils.write_tokens(source, source_path)
    test_utils.write_tokens(target, target_path)

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


@pytest.mark.parametrize("as_file_object", [True, False])
def test_load_model_from_memory(as_file_object):
    model_path = _get_model_path()
    files = {}

    for filename in os.listdir(model_path):
        with open(os.path.join(model_path, filename), "rb") as model_file:
            content = model_file.read()
            if as_file_object:
                content = io.BytesIO(content)
            files[filename] = content

    translator = ctranslate2.Translator("aren-transliteration", files=files)

    def _translate():
        output = translator.translate_batch([["آ", "ت", "ز", "م", "و", "ن"]])
        assert output[0].hypotheses[0] == ["a", "t", "z", "m", "o", "n"]

    if as_file_object:
        for handle in files.values():
            handle.close()

    _translate()

    translator.unload_model()
    translator.load_model()

    _translate()


@test_utils.only_on_linux
def test_logging():
    import wurlitzer

    assert ctranslate2.get_log_level() == logging.WARNING

    ctranslate2.set_log_level(logging.INFO)
    assert ctranslate2.get_log_level() == logging.INFO

    with wurlitzer.pipes() as (_, err):
        _get_transliterator()
    assert err.read()

    ctranslate2.set_log_level(logging.WARNING)
    assert ctranslate2.get_log_level() == logging.WARNING

    with wurlitzer.pipes() as (_, err):
        _get_transliterator()
    assert not err.read()
