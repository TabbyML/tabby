import collections

from typing import Iterable, List, Optional

from ctranslate2.translator import (
    GenerationResult,
    Generator,
    ScoringResult,
    TranslationResult,
    Translator,
)


def register_extensions():
    """Registers additional attributes to compiled modules."""
    setattr(Translator, "translate_iterable", translator_translate_iterable)
    setattr(Translator, "score_iterable", translator_score_iterable)
    setattr(Generator, "generate_iterable", generator_generate_iterable)
    setattr(Generator, "score_iterable", generator_score_iterable)


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
      An iterable of :class:`ctranslate2.TranslationResult` instances.
    """
    yield from _process_iterable(
        translator.translate_batch,
        source if target_prefix is None else zip(source, target_prefix),
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
      An iterable of :class:`ctranslate2.ScoringResult` instances.
    """
    yield from _process_iterable(
        translator.score_batch,
        zip(source, target),
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
      An iterable of :class:`ctranslate2.GenerationResult` instances.
    """
    yield from _process_iterable(
        generator.generate_batch,
        start_tokens,
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
      An iterable of :class:`ctranslate2.ScoringResult` instances.
    """
    yield from _process_iterable(
        generator.score_batch,
        tokens,
        max_batch_size,
        batch_type,
        **kwargs,
    )


def _process_iterable(process_func, iterable, max_batch_size, batch_type, **kwargs):
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be >= 1")

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
