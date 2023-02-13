import os
import shutil

import numpy as np
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
        dict(),
    ),
    (
        "Helsinki-NLP/opus-mt-en-roa",
        ">>ind<< ▁The ▁Prime ▁Minister ▁is ▁coming ▁back ▁tomorrow . </s>",
        "",
        "▁Per da na ▁Men teri ▁akan ▁kembali ▁besok .",
        dict(),
    ),
    (
        "Helsinki-NLP/opus-mt-mul-en",
        "▁Bon jo ur ▁le ▁mo nde </s>",
        "",
        "▁Welcome ▁to ▁the ▁World",
        dict(),
    ),
    (
        "facebook/m2m100_418M",
        "__en__ ▁Hello ▁world ! </s>",
        "__de__",
        "__de__ ▁Hallo ▁der ▁Welt !",
        dict(),
    ),
    (
        "facebook/mbart-large-50-many-to-many-mmt",
        "en_XX ▁Hello ▁world ! </s>",
        "de_DE",
        "de_DE ▁Hallo ▁Welt !",
        dict(),
    ),
    (
        "facebook/mbart-large-en-ro",
        "▁UN ▁Chief ▁Say s ▁There ▁Is ▁No ▁Militar y ▁Solution ▁in ▁Syria </s> en_XX",
        "ro_RO",
        "▁Şe ful ▁ONU ▁de cla ră ▁că ▁nu ▁există ▁o ▁solu ţie ▁militar ă ▁în ▁Siria",
        dict(),
    ),
    (
        "facebook/bart-base",
        "<s> UN ĠChief ĠSays ĠThere ĠIs ĠNo <mask> Ġin ĠSyria </s>",
        "",
        "<s> UN ĠChief ĠSays ĠThere ĠIs ĠNo ĠWar Ġin ĠSyria",
        dict(),
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
        dict(length_penalty=0.6),
    ),
    (
        "facebook/nllb-200-distilled-600M",
        ["▁Hello ▁world ! </s> eng_Latn", "</s> eng_Latn"],
        ["fra_Latn", "fra_Latn"],
        ["fra_Latn ▁Bon jour ▁le ▁monde ▁!", "fra_Latn"],
        dict(),
    ),
    (
        "t5-small",
        "▁translate ▁English ▁to ▁German : ▁The ▁house ▁is ▁wonderful . </s>",
        "",
        "▁Das ▁Haus ▁ist ▁wunderbar .",
        dict(),
    ),
    (
        "ml6team/mt5-small-german-query-generation",
        "▁Das ▁Lama ▁( L ama ▁glam a ) ▁ist ▁eine ▁Art ▁der ▁Kam ele . "
        "▁Es ▁ist ▁in ▁den ▁süd amerikanische n ▁And en ▁ver breite t ▁und "
        "▁eine ▁vom ▁Guan ako ▁ab sta mmende ▁ Haustier form . </s>",
        "",
        "▁Was ▁ist ▁Lama ▁glam a ?",
        dict(),
    ),
]


@test_utils.only_on_linux
@pytest.mark.parametrize(
    "model,source_tokens,target_tokens,expected_tokens,kwargs",
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
    kwargs,
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
        **kwargs,
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
@test_utils.on_available_devices
@pytest.mark.parametrize("return_log_probs", [True, False])
@pytest.mark.parametrize("tensor_input", [True, False])
def test_transformers_lm_forward(tmpdir, device, return_log_probs, tensor_input):
    import torch
    import transformers

    model_name = "gpt2"

    model = transformers.GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    converter = ctranslate2.converters.TransformersConverter(model_name)
    output_dir = str(tmpdir.join("ctranslate2_model"))
    output_dir = converter.convert(output_dir)
    generator = ctranslate2.Generator(output_dir, device=device)

    text = ["Hello world!"]

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        inputs.to(device)
        model.to(device)
        output = model(**inputs)
        ref_output = output.logits
        if return_log_probs:
            ref_output = torch.nn.functional.log_softmax(ref_output, dim=-1)
        ref_output = ref_output.cpu().numpy()

    kwargs = dict(return_log_probs=return_log_probs)

    if tensor_input:
        inputs = tokenizer(text, return_length=True, return_tensors="pt")
        inputs.to(device)
        ids = inputs.input_ids.to(torch.int32)
        lengths = inputs.length.to(torch.int32)

        if device == "cpu":
            ids = ids.numpy()
            lengths = lengths.numpy()

        ids = ctranslate2.StorageView.from_array(ids)
        lengths = ctranslate2.StorageView.from_array(lengths)

        with pytest.raises(ValueError, match="lengths"):
            generator.forward_batch(ids, **kwargs)
        output = generator.forward_batch(ids, lengths, **kwargs)

    else:
        ids = tokenizer(text).input_ids
        output = generator.forward_batch(ids, **kwargs)

    if device == "cpu":
        output = np.array(output)
    else:
        output = torch.as_tensor(output, device=device).cpu().numpy()

    assert output.shape == ref_output.shape
    np.testing.assert_allclose(output, ref_output, rtol=1e-2)


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


@test_utils.only_on_linux
def test_transformers_generator_suppress_sequences(tmpdir):
    converter = ctranslate2.converters.TransformersConverter("gpt2")
    output_dir = str(tmpdir.join("ctranslate2_model"))
    output_dir = converter.convert(output_dir)
    generator = ctranslate2.Generator(output_dir)

    output = generator.generate_batch(
        [["<|endoftext|>"]],
        max_length=10,
        suppress_sequences=[["Ġfirst", "Ġtime"]],
    )

    expected_tokens = "Ċ The Ġfirst Ġof Ġthe Ġthree Ġnew Ġseries Ġof Ġthe".split()
    assert output[0].sequences[0] == expected_tokens


@test_utils.only_on_linux
@test_utils.on_available_devices
@pytest.mark.parametrize(
    "prompts,expected_transcriptions,expected_no_speech_probs",
    [
        (
            [
                [
                    "<|startoftranscript|>",
                    "<|en|>",
                    "<|transcribe|>",
                    "<|notimestamps|>",
                ],
                [
                    "<|startoftranscript|>",
                    "<|en|>",
                    "<|transcribe|>",
                    "<|notimestamps|>",
                ],
            ],
            [
                " Mr. Quilter is the apostle of the middle classes and we are glad"
                " to welcome his gospel.",
                " And so my fellow Americans ask not what your country can do for you,"
                " ask what you can do for your country.",
            ],
            [
                pytest.approx(0.0022832120303064585, abs=1e-4),
                pytest.approx(0.06885894387960434, abs=1e-3),
            ],
        ),
        (
            [
                ["<|startoftranscript|>", "<|en|>", "<|transcribe|>"],
                ["<|startoftranscript|>", "<|en|>", "<|transcribe|>"],
            ],
            [
                " Mr. Quilter is the apostle of the middle classes and we are glad"
                " to welcome his gospel.",
                " And so, my fellow Americans, ask not what your country can do for you,"
                " ask what you can do for your country.",
            ],
            [
                pytest.approx(0.0022832120303064585, abs=1e-4),
                pytest.approx(0.06885894387960434, abs=1e-3),
            ],
        ),
        (
            [
                [
                    "<|startoftranscript|>",
                    "<|en|>",
                    "<|transcribe|>",
                    "<|notimestamps|>",
                ],
                [
                    "<|startofprev|>",
                    "<|startoftranscript|>",
                    "<|en|>",
                    "<|transcribe|>",
                ],
            ],
            [
                " Mr. Quilter is the apostle of the middle classes and we are glad"
                " to welcome his gospel.",
                " And so my fellow Americans ask not what your country can do for you,"
                " ask what you can do for your country.",
            ],
            [
                pytest.approx(0.0022832120303064585, abs=1e-4),
                pytest.approx(0.06885894387960434, abs=1e-2),
            ],
        ),
    ],
)
def test_transformers_whisper(
    tmpdir, device, prompts, expected_transcriptions, expected_no_speech_probs
):
    import transformers

    model_name = "openai/whisper-tiny"
    converter = ctranslate2.converters.TransformersConverter(model_name)
    output_dir = str(tmpdir.join("ctranslate2_model"))
    output_dir = converter.convert(output_dir)

    audio_paths = [
        os.path.join(test_utils.get_data_dir(), "audio", "mr_quilter.npy"),
        os.path.join(test_utils.get_data_dir(), "audio", "jfk.npy"),
    ]
    audio = list(map(np.load, audio_paths))

    processor = transformers.WhisperProcessor.from_pretrained(model_name)

    def _get_features(audio):
        # Pad after computing the log-Mel spectrogram to match the openai/whisper behavior.
        inputs = processor(audio, padding=False, sampling_rate=16000)
        features = inputs.input_features[0]
        features = np.pad(features, [(0, 0), (0, 3000 - features.shape[-1])])
        return features

    features = np.stack(list(map(_get_features, audio)))
    features = ctranslate2.StorageView.from_array(features)

    model = ctranslate2.models.Whisper(output_dir, device=device)

    for result in model.detect_language(features):
        best_lang, best_prob = result[0]
        assert best_lang == "<|en|>"
        assert best_prob > 0.9

    results = model.generate(
        features,
        prompts,
        beam_size=2,
        num_hypotheses=2,
        return_no_speech_prob=True,
    )

    timestamp_begin = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>") + 1

    for prompt, result, expected_transcription, expected_no_speech_prob in zip(
        prompts, results, expected_transcriptions, expected_no_speech_probs
    ):
        assert len(result.sequences_ids) == 2
        assert result.no_speech_prob == expected_no_speech_prob

        for tokens in result.sequences_ids:
            if "<|notimestamps|>" in prompt:
                assert all(token < timestamp_begin for token in tokens)
            else:
                assert tokens[0] >= timestamp_begin
                assert tokens[-1] >= timestamp_begin
                assert tokens[-1] > tokens[0]

        token_ids = list(
            filter(lambda token: token < timestamp_begin, result.sequences_ids[0])
        )

        transcription = processor.decode(token_ids)
        assert transcription == expected_transcription


@test_utils.only_on_linux
def test_transformers_whisper_invalid_shape(tmpdir):
    import transformers

    model_name = "openai/whisper-tiny"
    converter = ctranslate2.converters.TransformersConverter(model_name)
    output_dir = str(tmpdir.join("ctranslate2_model"))
    output_dir = converter.convert(output_dir)

    audio_path = os.path.join(test_utils.get_data_dir(), "audio", "jfk.npy")
    audio = np.load(audio_path)

    processor = transformers.WhisperProcessor.from_pretrained(model_name)
    inputs = processor(audio, padding=False, return_tensors="np", sampling_rate=16000)
    features = ctranslate2.StorageView.from_array(inputs.input_features)

    model = ctranslate2.models.Whisper(output_dir)

    with pytest.raises(ValueError) as exception_info:
        model.detect_language(features)

    error_message = str(exception_info.value)
    assert "(1, 80, 3000)" in error_message
    assert "(1, 80, 1100)" in error_message
