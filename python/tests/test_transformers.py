import os
import shutil

import pytest
import test_utils

import ctranslate2


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


@test_utils.only_on_linux
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


@test_utils.only_on_linux
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


@test_utils.only_on_linux
def test_transformers_marianmt_vocabulary(clear_transformers_cache, tmpdir):
    converter = ctranslate2.converters.TransformersConverter(
        "Helsinki-NLP/opus-mt-en-de"
    )
    output_dir = str(tmpdir.join("ctranslate2_model"))
    output_dir = converter.convert(output_dir)

    with open(os.path.join(output_dir, "shared_vocabulary.txt")) as vocab_file:
        vocab = list(line.rstrip("\n") for line in vocab_file)

    assert vocab[-1] != "<pad>"


@test_utils.only_on_linux
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


@test_utils.only_on_linux
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


@test_utils.only_on_linux
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
