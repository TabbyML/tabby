import inspect
import json
import os
import shutil

import numpy as np
import pytest
import test_utils

import ctranslate2


@pytest.fixture
def clear_transformers_cache():
    """Clears the Transformers model cache after each test when running in a CI."""
    yield
    clear_transformers_cache_in_ci()


def clear_transformers_cache_in_ci():
    import transformers

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
    tmp_dir,
    model,
    source_tokens,
    target_tokens,
    expected_tokens,
    kwargs,
):
    converter = ctranslate2.converters.TransformersConverter(model)
    output_dir = str(tmp_dir.join("ctranslate2_model"))
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
        "bigcode/tiny_starcoder_py",
        (
            "<fim_prefix> def Ġprint _ one _ two _ three (): ĊĠĠĠ Ġprint (' one ') "
            "ĊĠĠĠĠ <fim_suffix> ĊĠĠĠ Ġprint (' three ') <fim_middle>"
        ),
        26,
        (
            "<fim_prefix> def Ġprint _ one _ two _ three (): ĊĠĠĠ Ġprint (' one ') "
            "ĊĠĠĠĠ <fim_suffix> ĊĠĠĠ Ġprint (' three ') <fim_middle>"
            " print (' two ')"
        ),
    ),
    (
        "Salesforce/codegen-350M-mono",
        "def Ġhello _ name ( name ):",
        25,
        "def Ġhello _ name ( name ):"
        ' Ċ      print ( f " Hello Ġ{ name } ") Ċ Ċ hello _ name (" John ")',
    ),
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
    (
        "bigscience/bloom-560m",
        "Hello , ĠI Ġam",
        20,
        "Hello , ĠI Ġam Ġa Ġnew bie Ġin Ġthe Ġworld Ġof Ġweb Ġdesign Ġand ĠI Ġam "
        "Ġlooking Ġfor Ġa Ġweb Ġdeveloper",
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
    tmp_dir,
    model,
    start_tokens,
    max_length,
    expected_tokens,
):
    converter = ctranslate2.converters.TransformersConverter(model)
    output_dir = str(tmp_dir.join("ctranslate2_model"))
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
def test_transformers_dtype(clear_transformers_cache, tmp_dir):
    converter = ctranslate2.converters.TransformersConverter("facebook/opt-350m")
    output_dir = str(tmp_dir.join("ctranslate2_model"))
    output_dir = converter.convert(output_dir)

    model_b = os.path.getsize(os.path.join(output_dir, "model.bin"))
    model_mb = model_b / (1000**2)

    assert model_mb < 700


@test_utils.only_on_linux
def test_transformers_marianmt_vocabulary(clear_transformers_cache, tmp_dir):
    converter = ctranslate2.converters.TransformersConverter(
        "Helsinki-NLP/opus-mt-en-de"
    )
    output_dir = str(tmp_dir.join("ctranslate2_model"))
    output_dir = converter.convert(output_dir)

    vocabulary_path = os.path.join(output_dir, "shared_vocabulary.json")
    with open(vocabulary_path, encoding="utf-8") as vocabulary_file:
        vocab = json.load(vocabulary_file)

    assert vocab[-1] != "<pad>"


@test_utils.only_on_linux
@pytest.mark.parametrize("beam_size", [1, 2])
def test_transformers_marianmt_disable_unk(
    clear_transformers_cache, tmp_dir, beam_size
):
    converter = ctranslate2.converters.TransformersConverter(
        "Helsinki-NLP/opus-mt-en-roa"
    )
    output_dir = str(tmp_dir.join("ctranslate2_model"))
    output_dir = converter.convert(output_dir)

    tokens = ">>ind<< ▁The ▁Prime <unk> ▁is ▁coming ▁back ▁tomorrow . </s>".split()
    translator = ctranslate2.Translator(output_dir)
    output = translator.translate_batch([tokens], beam_size=beam_size, disable_unk=True)
    assert "<unk>" not in output[0].hypotheses[0]


@test_utils.only_on_linux
@test_utils.on_available_devices
@pytest.mark.parametrize(
    "model_name",
    [
        "bert-base-uncased",
        "distilbert-base-uncased",
        "distilbert-base-cased-distilled-squad",
        "typeform/distilbert-base-uncased-mnli",
    ],
)
def test_transformers_encoder(clear_transformers_cache, tmp_dir, device, model_name):
    import torch
    import transformers

    text = ["Hello world!", "Hello, my dog is cute"]

    model = transformers.AutoModel.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", padding=True)

    inputs.to(device)
    model.to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    mask = inputs.attention_mask.unsqueeze(-1).cpu().numpy()
    ref_last_hidden_state = outputs.last_hidden_state.cpu().numpy()
    ref_pooler_output = (
        outputs.pooler_output.cpu().numpy()
        if hasattr(outputs, "pooler_output")
        else None
    )

    converter = ctranslate2.converters.TransformersConverter(model_name)
    output_dir = str(tmp_dir.join("ctranslate2_model"))
    output_dir = converter.convert(output_dir)

    encoder = ctranslate2.Encoder(output_dir, device=device)

    ids = [tokenizer(t).input_ids for t in text]
    outputs = encoder.forward_batch(ids)

    last_hidden_state = _to_numpy(outputs.last_hidden_state, device)
    assert last_hidden_state.shape == ref_last_hidden_state.shape

    last_hidden_state *= mask
    ref_last_hidden_state *= mask
    np.testing.assert_array_almost_equal(
        last_hidden_state, ref_last_hidden_state, decimal=5
    )

    if ref_pooler_output is not None:
        pooler_output = _to_numpy(outputs.pooler_output, device)
        assert pooler_output.shape == ref_pooler_output.shape
        np.testing.assert_array_almost_equal(
            pooler_output, ref_pooler_output, decimal=5
        )


def _to_numpy(storage, device):
    import torch

    return (
        np.array(storage)
        if device == "cpu"
        else torch.as_tensor(storage, device=device).cpu().numpy()
    )


@test_utils.only_on_linux
def test_transformers_gptbigcode(clear_transformers_cache, tmp_dir):
    import transformers

    _check_generator_logits(
        tmp_dir,
        "hf-internal-testing/tiny-random-GPTBigCodeForCausalLM",
        transformers.GPTBigCodeForCausalLM,
        transformers.AutoTokenizer,
        "hello",
    )


def _check_generator_logits(
    tmp_dir, model_name, hf_model_class, hf_tokenizer_class, input_text
):
    import torch

    model = hf_model_class.from_pretrained(model_name)
    tokenizer = hf_tokenizer_class.from_pretrained(model_name)

    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    ref_logits = outputs.logits.numpy()

    converter = ctranslate2.converters.TransformersConverter(model_name)
    output_dir = str(tmp_dir.join("ctranslate2_model"))
    output_dir = converter.convert(output_dir)

    generator = ctranslate2.Generator(output_dir)
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(input_text))
    logits = generator.forward_batch([tokens])
    logits = np.array(logits)

    assert logits.shape == ref_logits.shape
    np.testing.assert_array_almost_equal(logits, ref_logits)


class TestGeneration:
    @classmethod
    def teardown_class(cls):
        clear_transformers_cache_in_ci()

    @test_utils.only_on_linux
    def test_transformers_lm_scoring(self, tmp_dir):
        converter = ctranslate2.converters.TransformersConverter("gpt2")
        output_dir = str(tmp_dir.join("ctranslate2_model"))
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
    def test_transformers_lm_forward(
        self, tmp_dir, device, return_log_probs, tensor_input
    ):
        import torch
        import transformers

        model_name = "gpt2"

        model = transformers.GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        converter = ctranslate2.converters.TransformersConverter(model_name)
        output_dir = str(tmp_dir.join("ctranslate2_model"))
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
    def test_transformers_generator_on_iterables(self, tmp_dir):
        converter = ctranslate2.converters.TransformersConverter("gpt2")
        output_dir = str(tmp_dir.join("ctranslate2_model"))
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
    def test_transformers_generator_suppress_sequences(self, tmp_dir):
        converter = ctranslate2.converters.TransformersConverter("gpt2")
        output_dir = str(tmp_dir.join("ctranslate2_model"))
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
    @pytest.mark.parametrize("beam_size", [1, 2])
    def test_transformers_generator_ignore_prompt(self, tmp_dir, beam_size):
        converter = ctranslate2.converters.TransformersConverter("gpt2")
        output_dir = str(tmp_dir.join("ctranslate2_model"))
        output_dir = converter.convert(output_dir)
        generator = ctranslate2.Generator(output_dir)

        max_length = 20
        tokens = "Ċ The Ġfirst Ġtime ĠI".split()

        result_wo_prompt = generator.generate_batch(
            [tokens],
            beam_size=beam_size,
            max_length=max_length - len(tokens),
            return_scores=True,
            include_prompt_in_result=False,
        )[0]

        result_w_prompt = generator.generate_batch(
            [tokens],
            beam_size=beam_size,
            max_length=max_length - 1,
            return_scores=True,
        )[0]

        assert len(result_w_prompt.sequences[0]) == max_length
        assert tokens + result_wo_prompt.sequences[0] == result_w_prompt.sequences[0]

        cum_score_wo_prompt = result_wo_prompt.scores[0] * (
            len(result_wo_prompt.sequences[0])
        )
        cum_score_w_prompt = result_w_prompt.scores[0] * (
            len(result_w_prompt.sequences[0]) - 1
        )

        assert cum_score_wo_prompt == pytest.approx(cum_score_w_prompt, abs=1e-3)

    @test_utils.only_on_linux
    @pytest.mark.parametrize("beam_size", [1, 2])
    def test_transformers_generator_ignore_prompt_batch(self, tmp_dir, beam_size):
        converter = ctranslate2.converters.TransformersConverter("gpt2")
        output_dir = str(tmp_dir.join("ctranslate2_model"))
        output_dir = converter.convert(output_dir)
        generator = ctranslate2.Generator(output_dir)

        new_tokens = 2
        prompt = [
            "Ċ The Ġfirst Ġtime ĠI".split(),
            "Ċ The Ġfirst".split(),
        ]

        results = generator.generate_batch(
            prompt,
            beam_size=beam_size,
            min_length=new_tokens,
            max_length=new_tokens,
            include_prompt_in_result=False,
        )

        for tokens, result in zip(prompt, results):
            assert len(result.sequences[0]) == new_tokens

    @test_utils.only_on_linux
    def test_transformers_generator_static_prompt(self, tmp_dir):
        converter = ctranslate2.converters.TransformersConverter("gpt2")
        output_dir = str(tmp_dir.join("ctranslate2_model"))
        output_dir = converter.convert(output_dir)
        generator = ctranslate2.Generator(output_dir)

        max_length = 20
        prompt = "Ċ The Ġfirst Ġtime ĠI".split()

        expected_result = generator.generate_batch(
            [prompt],
            max_length=max_length,
            include_prompt_in_result=False,
        )[0]

        result = generator.generate_batch(
            [[expected_result.sequences[0][0]]],
            max_length=max_length - 1,
            static_prompt=prompt,
        )[0]

        assert result.sequences[0] == expected_result.sequences[0]

        result = generator.generate_batch(
            [expected_result.sequences[0][:2]],
            max_length=max_length - 2,
            static_prompt=prompt,
            include_prompt_in_result=False,
        )[0]

        assert (
            expected_result.sequences[0][:2] + result.sequences[0]
            == expected_result.sequences[0]
        )

        batch_results = generator.generate_batch(
            [[expected_result.sequences[0][0]], [expected_result.sequences[0][0]]],
            max_length=max_length - 1,
            static_prompt=prompt,
        )

        assert batch_results[0].sequences[0] == expected_result.sequences[0]
        assert batch_results[1].sequences[0] == expected_result.sequences[0]

    @test_utils.only_on_linux
    @pytest.mark.parametrize("return_log_prob", [True, False])
    def test_transformers_generator_token_streaming(self, tmp_dir, return_log_prob):
        converter = ctranslate2.converters.TransformersConverter("gpt2")
        output_dir = str(tmp_dir.join("ctranslate2_model"))
        output_dir = converter.convert(output_dir)
        generator = ctranslate2.Generator(output_dir)

        max_length = 20
        prompt = "Ċ The Ġfirst Ġtime ĠI".split()

        expected_result = generator.generate_batch(
            [prompt],
            max_length=max_length,
            return_scores=True,
            include_prompt_in_result=False,
        )[0]

        step_results = generator.generate_tokens(
            prompt, max_length=max_length, return_log_prob=return_log_prob
        )

        assert inspect.isgenerator(step_results)

        tokens = []
        ids = []
        cum_log_probs = 0

        for step_result in step_results:
            assert isinstance(step_result, ctranslate2.GenerationStepResult)

            tokens.append(step_result.token)
            ids.append(step_result.token_id)

            if return_log_prob:
                cum_log_probs += step_result.log_prob
            else:
                assert step_result.log_prob is None

        assert tokens == expected_result.sequences[0]
        assert ids == expected_result.sequences_ids[0]

        if return_log_prob:
            assert cum_log_probs / len(ids) == pytest.approx(
                expected_result.scores[0], abs=1e-5
            )

    @test_utils.only_on_linux
    def test_transformers_generator_token_streaming_early_stop(self, tmp_dir):
        converter = ctranslate2.converters.TransformersConverter("gpt2")
        output_dir = str(tmp_dir.join("ctranslate2_model"))
        output_dir = converter.convert(output_dir)
        generator = ctranslate2.Generator(output_dir)

        prompt = "Ċ The Ġfirst Ġtime ĠI".split()
        results = generator.generate_tokens(prompt)
        for result in results:
            break

        results.close()


class TestWhisper:
    @classmethod
    def teardown_class(cls):
        clear_transformers_cache_in_ci()

    @test_utils.only_on_linux
    @test_utils.on_available_devices
    @pytest.mark.parametrize(
        "model_name,prompts,expected_transcriptions,expected_no_speech_probs",
        [
            (
                "openai/whisper-tiny",
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
                        "ĠAnd",
                        "Ġthus",
                        "Ġmy",
                    ],
                ],
                [
                    " Mr. Quilter is the apostle of the middle classes and we are glad"
                    " to welcome his gospel.",
                    " And thus my fellow Americans ask not what your country can do for you,"
                    " ask what you can do for your country.",
                ],
                [
                    pytest.approx(0.0022832120303064585, abs=1e-4),
                    pytest.approx(0.06885894387960434, abs=1e-3),
                ],
            ),
            (
                "openai/whisper-tiny",
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
                "openai/whisper-tiny.en",
                [["<|startoftranscript|>"], ["<|startoftranscript|>"]],
                [
                    " Mr. Quilter is the apostle of the middle classes, and we are glad"
                    " to welcome his gospel.",
                    " And so, my fellow Americans ask not what your country can do for you"
                    " ask what you can do for your country.",
                ],
                [
                    pytest.approx(0.02644546702504158, abs=1e-4),
                    pytest.approx(0.062380101531744, abs=1e-3),
                ],
            ),
        ],
    )
    def test_transformers_whisper(
        self,
        tmp_dir,
        device,
        model_name,
        prompts,
        expected_transcriptions,
        expected_no_speech_probs,
    ):
        import transformers

        converter = ctranslate2.converters.TransformersConverter(model_name)
        output_dir = str(tmp_dir.join("ctranslate2_model"))
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

        assert model.is_multilingual == (not model_name.endswith(".en"))

        if model.is_multilingual:
            for result in model.detect_language(features):
                best_lang, best_prob = result[0]
                assert best_lang == "<|en|>"
                assert best_prob > 0.9
        else:
            with pytest.raises(RuntimeError, match="multilingual"):
                model.detect_language(features)

        results = model.generate(
            features,
            prompts,
            beam_size=2,
            num_hypotheses=2,
            return_no_speech_prob=True,
        )

        timestamp_begin = (
            processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>") + 1
        )

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
    @test_utils.on_available_devices
    @pytest.mark.parametrize(
        "test_names", [["jfk"], ["jfk", "jfk"], ["mr_quilter", "jfk"]]
    )
    def test_transformers_whisper_align(self, tmp_dir, device, test_names):
        import transformers

        test_cases = []
        audio = []

        test_dir = os.path.join(test_utils.get_data_dir(), "audio")

        for name in test_names:
            audio_path = os.path.join(test_dir, "%s.npy" % name)
            audio.append(np.load(audio_path))

            test_case_path = os.path.join(test_dir, "%s_alignments.json" % name)
            with open(test_case_path) as test_case_file:
                test_cases.append(json.load(test_case_file))

        model_name = "openai/whisper-tiny.en"
        converter = ctranslate2.converters.TransformersConverter(model_name)
        output_dir = str(tmp_dir.join("ctranslate2_model"))
        output_dir = converter.convert(output_dir)

        processor = transformers.WhisperProcessor.from_pretrained(model_name)
        inputs = processor(audio, return_tensors="np", sampling_rate=16000)
        features = ctranslate2.StorageView.from_array(inputs.input_features)

        model = ctranslate2.models.Whisper(output_dir, device=device)

        results = model.align(
            features,
            [50257],
            [test_case["text_tokens"] for test_case in test_cases],
            [test_case["num_frames"] for test_case in test_cases],
        )

        for result, test_case in zip(results, test_cases):
            assert np.sum(result.text_token_probs) == pytest.approx(
                test_case["expected_text_token_probs_sum"], abs=1e-3
            )

            assert result.alignments == [
                tuple(pair) for pair in test_case["expected_alignments"]
            ]

    @test_utils.only_on_linux
    @test_utils.on_available_devices
    def test_transformers_whisper_encode(self, tmp_dir, device):
        import transformers

        model_name = "openai/whisper-tiny.en"
        converter = ctranslate2.converters.TransformersConverter(model_name)
        output_dir = str(tmp_dir.join("ctranslate2_model"))
        output_dir = converter.convert(output_dir)

        audio_path = os.path.join(test_utils.get_data_dir(), "audio", "jfk.npy")
        audio = np.load(audio_path)

        processor = transformers.WhisperProcessor.from_pretrained(model_name)
        inputs = processor(audio, sampling_rate=16000)
        features = inputs.input_features[0]
        features = np.expand_dims(features, 0)
        features = ctranslate2.StorageView.from_array(features)

        model = ctranslate2.models.Whisper(output_dir, device=device)
        encoded = model.encode(features)
        prompts = [["<|startoftranscript|>", "<|notimestamps|>"]]
        result = model.generate(encoded, prompts)[0]

        transcription = processor.decode(result.sequences_ids[0])

        assert transcription == (
            " And so my fellow Americans ask not what your country can do for you, "
            "ask what you can do for your country."
        )

    @test_utils.only_on_linux
    def test_transformers_whisper_partial_audio_context(self, tmp_dir):
        import transformers

        model_name = "openai/whisper-tiny"
        converter = ctranslate2.converters.TransformersConverter(model_name)
        output_dir = str(tmp_dir.join("ctranslate2_model"))
        output_dir = converter.convert(output_dir)

        audio_path = os.path.join(test_utils.get_data_dir(), "audio", "jfk.npy")
        audio = np.load(audio_path)

        processor = transformers.WhisperProcessor.from_pretrained(model_name)
        inputs = processor(
            audio, padding=False, return_tensors="np", sampling_rate=16000
        )
        features = ctranslate2.StorageView.from_array(inputs.input_features)

        model = ctranslate2.models.Whisper(output_dir)
        encoder_output = model.encode(features)

        assert encoder_output.shape == [1, features.shape[2] // 2, 384]

    @test_utils.only_on_linux
    def test_transformers_whisper_include_tokenizer_json(self, tmp_dir):
        model_name = "openai/whisper-tiny"
        converter = ctranslate2.converters.TransformersConverter(
            model_name, copy_files=["tokenizer.json"]
        )
        output_dir = str(tmp_dir.join("ctranslate2_model"))
        output_dir = converter.convert(output_dir)
        assert os.path.isfile(os.path.join(output_dir, "tokenizer.json"))


class TestWav2Vec2:
    @classmethod
    def teardown_class(cls):
        clear_transformers_cache_in_ci()

    @test_utils.only_on_linux
    @test_utils.on_available_devices
    @pytest.mark.parametrize(
        "model_name,expected_transcription",
        [
            (
                "facebook/wav2vec2-large-robust-ft-swbd-300h",
                [
                    "MISTER QUILTER IS THE APOSSEL OF THE MIDDLE CLASSES AND"
                    " WE ARE GLAD TO WELCOME HIS GOSPEL",
                ],
            ),
        ],
    )
    def test_transformers_wav2vec2(
        self,
        tmp_dir,
        device,
        model_name,
        expected_transcription,
    ):
        import torch
        import transformers

        converter = ctranslate2.converters.TransformersConverter(
            model_name, load_as_float16="int8"
        )
        output_dir = str(tmp_dir.join("ctranslate2_model"))
        output_dir = converter.convert(output_dir)
        # 24 x Wav2Vec2EncoderLayerStableLayerNorm converted & saved

        w2v2_model = transformers.Wav2Vec2ForCTC.from_pretrained(model_name)
        del w2v2_model.wav2vec2.encoder.layers
        del w2v2_model.wav2vec2.encoder.layer_norm
        w2v2_model.save_pretrained(output_dir + "/wav2vec2_partial.bin")
        w2v2_processor = transformers.Wav2Vec2Processor.from_pretrained(model_name)
        torch.save(w2v2_processor, output_dir + "/wav2vec2_processor.bin")

        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        cpu_threads = int(os.environ.get("OMP_NUM_THREADS", 0))
        w2v2_model = transformers.Wav2Vec2ForCTC.from_pretrained(
            output_dir + "/wav2vec2_partial.bin"
        ).to(device)
        del w2v2_model.wav2vec2.encoder.layers
        del w2v2_model.wav2vec2.encoder.layer_norm
        w2v2_processor = torch.load(output_dir + "/wav2vec2_processor.bin")
        ct2_w2v2_model = ctranslate2.models.Wav2Vec2(
            output_dir,
            device=device,
            device_index=[0],
            compute_type="int8",
            intra_threads=cpu_threads,
            inter_threads=1,
        )

        speech_array = np.load(
            os.path.join(test_utils.get_data_dir(), "audio", "mr_quilter.npy")
        )
        input_values = w2v2_processor(
            speech_array,
            padding=True,
            return_tensors="pt",
            sampling_rate=16000,
        ).input_values

        with torch.no_grad():
            extract_features = w2v2_model.wav2vec2.feature_extractor(
                input_values.to(w2v2_model.device)
            ).transpose(1, 2)
            hidden_states, extract_features = w2v2_model.wav2vec2.feature_projection(
                extract_features
            )
            position_embeddings = w2v2_model.wav2vec2.encoder.pos_conv_embed(
                hidden_states
            )
            hidden_states = position_embeddings + hidden_states
            # hidden_states = w2v2_model.encoder.dropout(hidden_states)
            # Dropout(p=0.0, inplace=False) bypassed

        if ct2_w2v2_model.device == "cuda":
            hidden_states = hidden_states.cpu()
        else:
            hidden_states.numpy()

        hidden_states = np.ascontiguousarray(hidden_states)
        hidden_states = ctranslate2.StorageView.from_array(hidden_states)
        to_cpu = (
            ct2_w2v2_model.device == "cuda" and len(ct2_w2v2_model.device_index) > 1
        )
        ct2_output = ct2_w2v2_model.encode(
            hidden_states,
            to_cpu=to_cpu,
        )  # 24 x Wav2Vec2EncoderLayerStableLayerNorm processed
        if ct2_w2v2_model.device == "cuda":
            hidden_states = torch.as_tensor(
                ct2_output,
                device=ct2_w2v2_model.device,
            )
        else:
            hidden_states = torch.as_tensor(
                np.array(ct2_output),
                dtype=torch.float32,
                device=ct2_w2v2_model.device,
            )

        encoder_outputs = transformers.modeling_outputs.BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=None,
            attentions=None,
        )
        hidden_states = encoder_outputs[0]
        outputs = transformers.modeling_outputs.Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        hidden_states = outputs[0]
        # hidden_states = w2v2_model.dropout(hidden_states)
        # Dropout(p=0.0, inplace=False) bypassed

        with torch.no_grad():
            logits = w2v2_model.lm_head(hidden_states.to(torch.float32))[0]

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = w2v2_processor.decode(predicted_ids, output_word_offsets=True)

        assert transcription[0] == expected_transcription[0]
