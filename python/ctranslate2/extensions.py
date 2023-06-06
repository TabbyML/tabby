import collections
import itertools
import queue
import threading

from typing import Iterable, List, Optional, Union

from ctranslate2._ext import (
    GenerationResult,
    GenerationStepResult,
    Generator,
    ScoringResult,
    TranslationResult,
    Translator,
)


def register_extensions():
    """Registers additional attributes to compiled modules."""
    setattr(Translator, "translate_iterable", translator_translate_iterable)
    setattr(Translator, "score_iterable", translator_score_iterable)
    setattr(Translator, "generate_tokens", translator_generate_tokens)
    setattr(Generator, "generate_iterable", generator_generate_iterable)
    setattr(Generator, "score_iterable", generator_score_iterable)
    setattr(Generator, "generate_tokens", generator_generate_tokens)


def translator_translate_iterable(
    translator: Translator,
    source: Iterable[List[str]],
    target_prefix: Optional[Iterable[List[str]]] = None,
    max_batch_size: int = 32,
    batch_type: str = "examples",
    **kwargs,
) -> Iterable[TranslationResult]:
    """Translates an iterable of tokens.

    This method is built on top of :meth:`ctranslate2.Translator.translate_batch`
    to efficiently translate an arbitrarily large stream of data. It enables the
    following optimizations:

    * stream processing (the iterable is not fully materialized in memory)
    * parallel translations (if the translator has multiple workers)
    * asynchronous batch prefetching
    * local sorting by length

    Arguments:
      source: An iterable on source tokens.
      target_prefix: An optional iterable on target tokens used as prefix.
      max_batch_size: The maximum batch size.
      batch_type: Whether :obj:`max_batch_size` is the number of "examples" or "tokens".
      **kwargs: Any translation options accepted by
        :meth:`ctranslate2.Translator.translate_batch`.

    Returns:
      A generator iterator over :class:`ctranslate2.TranslationResult` instances.
    """
    iterables = [source]
    if target_prefix is not None:
        iterables.append(target_prefix)

    yield from _process_iterable(
        translator.translate_batch,
        iterables,
        max_batch_size,
        batch_type,
        **kwargs,
    )


def translator_score_iterable(
    translator: Translator,
    source: Iterable[List[str]],
    target: Iterable[List[str]],
    max_batch_size: int = 64,
    batch_type: str = "examples",
    **kwargs,
) -> Iterable[ScoringResult]:
    """Scores an iterable of tokens.

    This method is built on top of :meth:`ctranslate2.Translator.score_batch`
    to efficiently score an arbitrarily large stream of data. It enables the
    following optimizations:

    * stream processing (the iterable is not fully materialized in memory)
    * parallel scoring (if the translator has multiple workers)
    * asynchronous batch prefetching
    * local sorting by length

    Arguments:
      source: An iterable on source tokens.
      target: An iterable on target tokens.
      max_batch_size: The maximum batch size.
      batch_type: Whether :obj:`max_batch_size` is the number of "examples" or "tokens".
      **kwargs: Any scoring options accepted by
        :meth:`ctranslate2.Translator.score_batch`.

    Returns:
      A generator iterator over :class:`ctranslate2.ScoringResult` instances.
    """
    yield from _process_iterable(
        translator.score_batch,
        [source, target],
        max_batch_size,
        batch_type,
        **kwargs,
    )


def generator_generate_iterable(
    generator: Generator,
    start_tokens: Iterable[List[str]],
    max_batch_size: int = 32,
    batch_type: str = "examples",
    **kwargs,
) -> Iterable[GenerationResult]:
    """Generates from an iterable of start tokens.

    This method is built on top of :meth:`ctranslate2.Generator.generate_batch`
    to efficiently run generation on an arbitrarily large stream of data. It enables
    the following optimizations:

    * stream processing (the iterable is not fully materialized in memory)
    * parallel generations (if the generator has multiple workers)
    * asynchronous batch prefetching
    * local sorting by length

    Arguments:
      start_tokens: An iterable on start tokens.
      max_batch_size: The maximum batch size.
      batch_type: Whether :obj:`max_batch_size` is the number of "examples" or "tokens".
      **kwargs: Any generation options accepted by
        :meth:`ctranslate2.Generator.generate_batch`.

    Returns:
      A generator iterator over :class:`ctranslate2.GenerationResult` instances.
    """
    yield from _process_iterable(
        generator.generate_batch,
        [start_tokens],
        max_batch_size,
        batch_type,
        **kwargs,
    )


def generator_score_iterable(
    generator: Generator,
    tokens: Iterable[List[str]],
    max_batch_size: int = 64,
    batch_type: str = "examples",
    **kwargs,
) -> Iterable[ScoringResult]:
    """Scores an iterable of tokens.

    This method is built on top of :meth:`ctranslate2.Generator.score_batch`
    to efficiently score an arbitrarily large stream of data. It enables
    the following optimizations:

    * stream processing (the iterable is not fully materialized in memory)
    * parallel scoring (if the generator has multiple workers)
    * asynchronous batch prefetching
    * local sorting by length

    Arguments:
      tokens: An iterable on tokens.
      max_batch_size: The maximum batch size.
      batch_type: Whether :obj:`max_batch_size` is the number of "examples" or "tokens".
      **kwargs: Any score options accepted by
        :meth:`ctranslate2.Generator.score_batch`.

    Returns:
      A generator iterator over :class:`ctranslate2.ScoringResult` instances.
    """
    yield from _process_iterable(
        generator.score_batch,
        [tokens],
        max_batch_size,
        batch_type,
        **kwargs,
    )


def translator_generate_tokens(
    translator: Translator,
    source: List[str],
    target_prefix: Optional[List[str]] = None,
    *,
    max_decoding_length: int = 256,
    min_decoding_length: int = 1,
    sampling_topk: int = 1,
    sampling_temperature: float = 1,
    return_log_prob: bool = False,
    repetition_penalty: float = 1,
    no_repeat_ngram_size: int = 0,
    disable_unk: bool = False,
    suppress_sequences: Optional[List[List[str]]] = None,
    end_token: Optional[Union[str, List[str], List[int]]] = None,
    max_input_length: int = 1024,
    use_vmap: bool = False,
) -> Iterable[GenerationStepResult]:
    """Yields tokens as they are generated by the model.

    Arguments:
      source: Source tokens.
      target_prefix: Optional target prefix tokens.
      max_decoding_length: Maximum prediction length.
      min_decoding_length: Minimum prediction length.
      sampling_topk: Randomly sample predictions from the top K candidates.
      sampling_temperature: Sampling temperature to generate more random samples.
      return_log_prob: Include the token log probability in the result.
      repetition_penalty: Penalty applied to the score of previously generated tokens
        (set > 1 to penalize).
      no_repeat_ngram_size: Prevent repetitions of ngrams with this size
        (set 0 to disable).
      disable_unk: Disable the generation of the unknown token.
      suppress_sequences: Disable the generation of some sequences of tokens.
      end_token: Stop the decoding on one of these tokens (defaults to the model EOS token).
      max_input_length: Truncate inputs after this many tokens (set 0 to disable).
      use_vmap: Use the vocabulary mapping file saved in this model

    Returns:
      A generator iterator over :class:`ctranslate2.GenerationStepResult` instances.

    Note:
      This generation method is not compatible with beam search which requires a complete decoding.
    """
    yield from _generate_tokens(
        translator.translate_batch,
        [source],
        [target_prefix] if target_prefix is not None else None,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        disable_unk=disable_unk,
        suppress_sequences=suppress_sequences,
        end_token=end_token,
        max_decoding_length=max_decoding_length,
        min_decoding_length=min_decoding_length,
        sampling_topk=sampling_topk,
        sampling_temperature=sampling_temperature,
        return_scores=return_log_prob,
        max_input_length=max_input_length,
        use_vmap=use_vmap,
    )


def generator_generate_tokens(
    generator: Generator,
    prompt: List[str],
    *,
    max_length: int = 512,
    min_length: int = 0,
    sampling_topk: int = 1,
    sampling_temperature: float = 1,
    return_log_prob: bool = False,
    repetition_penalty: float = 1,
    no_repeat_ngram_size: int = 0,
    disable_unk: bool = False,
    suppress_sequences: Optional[List[List[str]]] = None,
    end_token: Optional[Union[str, List[str], List[int]]] = None,
    static_prompt: Optional[List[str]] = None,
    cache_static_prompt: bool = True,
) -> Iterable[GenerationStepResult]:
    """Yields tokens as they are generated by the model.

    Arguments:
      prompt: The prompt tokens.
      max_length: Maximum generation length.
      min_length: Minimum generation length.
      sampling_topk: Randomly sample predictions from the top K candidates.
      sampling_temperature: Sampling temperature to generate more random samples.
      return_log_prob: Include the token log probability in the result.
      repetition_penalty: Penalty applied to the score of previously generated tokens
        (set > 1 to penalize).
      no_repeat_ngram_size: Prevent repetitions of ngrams with this size
        (set 0 to disable).
      disable_unk: Disable the generation of the unknown token.
      suppress_sequences: Disable the generation of some sequences of tokens.
      end_token: Stop the decoding on one these tokens (defaults to the model EOS token).
      static_prompt: If the model expects a static prompt (a.k.a. system prompt)
        it can be set here to simplify the inputs and optionally cache the model
        state for this prompt to accelerate future generations.
      cache_static_prompt: Cache the model state after the static prompt and
        reuse it for future generations using the same static prompt.

    Returns:
      A generator iterator over :class:`ctranslate2.GenerationStepResult` instances.

    Note:
      This generation method is not compatible with beam search which requires a complete decoding.
    """
    yield from _generate_tokens(
        generator.generate_batch,
        [prompt],
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        disable_unk=disable_unk,
        suppress_sequences=suppress_sequences,
        end_token=end_token,
        max_length=max_length,
        min_length=min_length,
        sampling_topk=sampling_topk,
        sampling_temperature=sampling_temperature,
        return_scores=return_log_prob,
        static_prompt=static_prompt,
        cache_static_prompt=cache_static_prompt,
        include_prompt_in_result=False,
    )


def _generate_tokens(process_func, *args, **kwargs):
    step_results = queue.Queue()

    def _callback(step_result):
        step_results.put(step_result)

        if step_result.is_last:
            step_results.put(None)

    kwargs.update(
        {
            "asynchronous": True,
            "beam_size": 1,
            "callback": _callback,
        }
    )

    async_result = process_func(*args, **kwargs)[0]

    def _catch_exception():
        try:
            async_result.result()
        except Exception as e:
            step_results.put(e)

    thread = threading.Thread(target=_catch_exception, daemon=True)
    thread.start()

    while True:
        step_result = step_results.get()

        if step_result is None:
            break

        if isinstance(step_result, Exception):
            raise step_result

        yield step_result

    # Wait for the job to terminate before exiting.
    thread.join()


def _process_iterable(process_func, iterables, max_batch_size, batch_type, **kwargs):
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be >= 1")

    if len(iterables) == 1:
        iterable = iterables[0]
    else:
        iterable = itertools.zip_longest(*iterables)

    kwargs.update(
        {
            "max_batch_size": max_batch_size,
            "batch_type": batch_type,
            "asynchronous": True,
        }
    )

    read_batch_size = max_batch_size * 16 if max_batch_size > 1 else max_batch_size
    queue = collections.deque()

    for streams in _batch_iterator(iterable, read_batch_size, batch_type):
        queue.extend(process_func(*streams, **kwargs))

        while queue and queue[0].done():
            yield queue.popleft().result()

    while queue:
        yield queue.popleft().result()


def _batch_iterator(iterable, batch_size, batch_type):
    streams = None
    cur_batch_size = 0

    for example in iterable:
        if not isinstance(example, tuple):
            example = (example,)

        if streams is None:
            streams = tuple([] for _ in example)
        for batch, element in zip(streams, example):
            if element is None and len(streams) > 1:
                raise ValueError("Input iterables do not have the same length")
            batch.append(element)

        if batch_type == "examples":
            cur_batch_size += 1
        elif batch_type == "tokens":
            cur_batch_size += len(example[0])
        else:
            raise ValueError("Invalid batch type %s" % batch_type)

        if cur_batch_size >= batch_size:
            yield streams
            streams = None
            cur_batch_size = 0

    if streams is not None:
        yield streams
