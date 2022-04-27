# Python

**Table of contents**

1. [Installation](#installation)
1. [Model conversion API](#model-conversion-api)
1. [Translation API](#translation-api)
1. [Generation API](#generation-api)
1. [Utilities API](#utilities-api)
1. [Additional information](#additional-information)

## Installation

See the [Installation](../README.md#installation) section in the main README.

## Model conversion API

```python
converter = ctranslate2.converters.OpenNMTTFConverter(
    model_spec: ModelSpec,   # Specification of the model to convert.
    src_vocab: Union[str, opennmt.data.Vocab, List[str]],  # Source vocabulary.
    tgt_vocab: Union[str, opennmt.data.Vocab, List[str]],  # Target vocabulary.
    model_path: str = None,  # Path to a OpenNMT-tf checkpoint (mutually exclusive with variables)
    variables: dict = None,  # Dict of variables name to value (mutually exclusive with model_path).
)

converter = ctranslate2.converters.OpenNMTPyConverter(
    model_path: str,         # Path to the OpenNMT-py model (.pt file).
)

converter = ctranslate2.converters.FairseqConverter(
    model_path: str,              # Path to the Fairseq model (.pt file).
    data_dir: str,                # Path to the Fairseq data directory.
    source_lang: str = None,      # Source language (may be required if not declared in the model).
    target_lang: str = None,      # Target language (may be required if not declared in the model).
    fixed_dictionary: str = None, # Path to the fixed dictionary for multilingual models.
    no_default_special_tokens: bool = False,  # Require all special tokens to be provided by the user.
    user_dir: str = None,         # Path to the user module containing custom extensions.
)

converter = ctranslate2.converters.MarianConverter(
    model_path: str,         # Path to the Marian model (.npz file).
    vocab_paths: List[str],  # Paths to the vocabularies (.yml files).
)

converter = ctranslate2.converters.OpusMTConverter(
    model_dir: str,  # Path to the OPUS-MT model directory.
)

converter = ctranslate2.converters.OpenAIGPT2Converter(
    model_dir: str,  # Path to the GPT-2 model directory.
)

output_dir = converter.convert(
    output_dir: str,          # Path to the output directory.
    vmap: str = None,         # Path to a vocabulary mapping file.
    quantization: str = None, # Weights quantization: "int8", "int8_float16", "int16", or "float16".
    force: bool = False,      # Override output_dir if it exists.
)
```

## Translation API

### Constructor

```python
translator = ctranslate2.Translator(
    model_path: str,                # Path to the CTranslate2 model directory.
    device: str = "cpu",            # The device to use: "cpu", "cuda", or "auto".
    *,

    # The device ID, or list of device IDs, where to place this translator on.
    device_index: Union[int, List[int]] = 0,

    # The computation type: "default", "auto", "int8", "int8_float16", "int16", "float16", or "float",
    # or a dict mapping a device to a computation type.
    compute_type: Union[str, Dict[str, str]] = "default",

    inter_threads: int = 1,         # Maximum number of parallel translations.
    intra_threads: int = 0,         # Number of OpenMP threads to use per translation (CPU only).
                                    # Set 0 to use a default value.

    max_queued_batches: int = 0,    # Maximum number of batches in the translation queue (set -1 for unlimited, 0 for an automatic value).
                                    # When the queue is full, future requests will block until a free slot is available.
)
```

### Properties

```python
translator.device              # Device this translator is running on.
translator.device_index        # List of device IDs where this translator is running on.
translator.num_translators     # Number of translators backing this instance.
translator.num_queued_batches  # Number of batches waiting to be translated.
translator.num_active_batches  # Number of batches waiting to be translated or currently in translation.
```

### Batch translation

```python
translator.translate_batch(
    source: list,                      # A list of list of string.
    target_prefix: list = None,        # An optional list of list of string.
    *,
    max_batch_size: int = 0,           # Maximum batch size to run the model on.
    batch_type: str = "examples",      # Whether max_batch_size is the number of examples or tokens.
    asynchronous: bool = False,        # Run the translation asynchronously.
    beam_size: int = 2,                # Beam size (set 1 to run greedy search).
    num_hypotheses: int = 1,           # Number of hypotheses to return (should be <= beam_size
                                       # unless return_alternatives is set).
    length_penalty: float = 0,         # Length penalty constant to use during beam search.
    coverage_penalty: float = 0,       # Converage penalty constant to use during beam search.
    repetition_penalty: float = 1,     # Penalty applied to the score of previously generated tokens (set > 1 to penalize).
    disable_unk: bool = False,         # Disable the generation of the unknown token.
    prefix_bias_beta: float = 0,       # Parameter for biasing translations towards given prefix.
    allow_early_exit: bool = True,     # Allow the beam search to exit early when the first beam finishes.
    max_input_length: int = 1024,      # Truncate inputs after this many tokens (set 0 to disable).
    max_decoding_length: int = 256,    # Maximum prediction length.
    min_decoding_length: int = 1,      # Minimum prediction length.
    use_vmap: bool = False,            # Use the vocabulary mapping file saved in this model.
    normalize_scores: bool = False,    # Normalize the score by the hypothesis length.
    return_scores: bool = False,       # Include the prediction scores in the output.
    return_attention: bool = False,    # Include the attention vectors in the output.
    return_alternatives: bool = False, # Return alternatives at the first unconstrained decoding position.
    sampling_topk: int = 1,            # Randomly sample predictions from the top K candidates.
    sampling_temperature: float = 1,   # Sampling temperature to generate more random samples.
    replace_unknowns: bool = False,    # Replace unknown target tokens by the source token with the highest attention.
) -> Union[List[ctranslate2.TranslationResult], List[ctranslate2.AsyncTranslationResult]]
```

The result `ctranslate2.TranslationResult` has the following properties:

* `hypotheses`
* `scores` (empty if `return_scores` is set to `False`)
* `attention` (empty if `return_attention` is set to `False`)

With `asynchronous=True`, the function returns a list of `ctranslate2.AsyncTranslationResult` instances. The actual `ctranslate2.TranslationResult` instance can be retrieved by calling `.result()` on the asynchronous wrapper.

Also see the [`TranslationOptions`](../include/ctranslate2/translation.h) structure for more details about the options.

### File translation

```python
translator.translate_file(
    source_path: str,               # Source file.
    output_path: str,               # Output file.
    target_path: str = None,        # Target prefix file.
    *,
    max_batch_size: int = 32,       # Maximum batch size to run the model on.
    read_batch_size: int = 0,       # Number of sentences to read at once.
    batch_type: str = "examples",   # Whether the batch size is the number of examples or tokens.
    beam_size: int = 2,
    num_hypotheses: int = 1,
    length_penalty: float = 0,
    coverage_penalty: float = 0,
    repetition_penalty: float = 1,
    disable_unk: bool = False,
    prefix_bias_beta: float = 0,
    allow_early_exit: bool = True,
    max_input_length: int = 1024,
    max_decoding_length: int = 256,
    min_decoding_length: int = 1,
    use_vmap: bool = False,
    normalize_scores: bool = False,
    with_scores: bool = False,
    sampling_topk: int = 1,
    sampling_temperature: float = 1,
    replace_unknowns: bool = False,
    source_tokenize_fn: callable = None,   # Function with signature: string -> list of strings
    target_tokenize_fn: callable = None,   # Function with signature: string -> list of strings
    target_detokenize_fn: callable = None, # Function with signature: list of strings -> string
) -> ctranslate2.TranslationStats
```

The returned statistics object has the following properties:

* `num_tokens`: the number of generated target tokens
* `num_examples`: the number of translated examples
* `total_time_in_ms`: the total translation time in milliseconds

### Batch scoring

```python
translator.score_batch(
    source: List[List[str]],
    target: List[List[str]],
    *,
    max_batch_size: int = 0,       # Maximum batch size to run the model on.
    batch_type: str = "examples",  # Whether max_batch_size is the number of examples or tokens.
    max_input_length: int = 1024,  # Truncate inputs after this many tokens (set 0 to disable).
) -> List[List[float]]
```

The returned score sequences include the score of the end of sentence token `</s>`.

### File scoring

```python
translator.score_file(
    source_path: str,              # Source file.
    target_path: str,              # Target file.
    output_path: str,              # Output file.
    *,
    max_batch_size: int = 32,      # Maximum batch size to run the model on.
    read_batch_size: int = 0,      # Number of sentences to read at once.
    batch_type: str = "examples",  # Whether the batch size is the number of examples or tokens.
    max_input_length: int = 1024,  # Truncate inputs after this many tokens (set 0 to disable).
    with_tokens_score: bool = False,       # Include token-level scores in the output.
    source_tokenize_fn: callable = None,   # Function with signature: string -> list of strings
    target_tokenize_fn: callable = None,   # Function with signature: string -> list of strings
    target_detokenize_fn: callable = None, # Function with signature: list of strings -> string
) -> ctranslate2.TranslationStats
```

Each line in `output_path` will have the format:

```text
<score> ||| <target>
```

The score is normalized by the target length which includes the end of sentence token `</s>`. The returned statistics object has the following properties:

* `num_tokens`: the number of scored target tokens
* `num_examples`: the number of scored examples
* `total_time_in_ms`: the total scoring time in milliseconds

### Memory management

* `translator.unload_model(to_cpu: bool = False)`<br/>Unload the model attached to this translator but keep enough runtime context to quickly resume translation on the initial device. The model is not guaranteed to be unloaded if translations are running concurrently.
  * `to_cpu`: If `True`, the model is moved to the CPU memory and not fully unloaded.
* `translator.load_model()`<br/>Load the model back to the initial device.
* `translator.model_is_loaded`<br/>Property set to `True` when the model is loaded on the initial device and ready to be used.
* `del translator`<br/>Release the translator resources.

## Generation API

### Constructor

```python
generator = ctranslate2.Generator(
    model_path: str,                # Path to the CTranslate2 model directory.
    device: str = "cpu",            # The device to use: "cpu", "cuda", or "auto".
    *,

    # The device ID, or list of device IDs, where to place this generator on.
    device_index: Union[int, List[int]] = 0,

    # The computation type: "default", "auto", "int8", "int8_float16", "int16", "float16", or "float",
    # or a dict mapping a device to a computation type.
    compute_type: Union[str, Dict[str, str]] = "default",

    inter_threads: int = 1,         # Maximum number of parallel generations.
    intra_threads: int = 0,         # Number of OpenMP threads to use per generator (CPU only).
                                    # Set 0 to use a default value.

    max_queued_batches: int = 0,    # Maximum number of batches in the generation queue (set -1 for unlimited, 0 for an automatic value).
                                    # When the queue is full, future requests will block until a free slot is available.
)
```

### Properties

```python
generator.device              # Device this generator is running on.
generator.device_index        # List of device IDs where this generator is running on.
generator.num_generators      # Number of generators backing this instance.
generator.num_queued_batches  # Number of batches waiting to be processed.
generator.num_active_batches  # Number of batches waiting to be processed or currently processed.
```

### Batch generation

If the decoder starts from a special start token like `<s>`, this token should be included in the start tokens.

```python
generator.generate_batch(
    start_tokens: List[List[str]],     # A list of list of string.
    *,
    max_batch_size: int = 0,           # Maximum batch size to run the model on.
    batch_type: str = "examples",      # Whether max_batch_size is the number of examples or tokens.
    asynchronous: bool = False,        # Run the generation asynchronously.
    beam_size: int = 1,                # Beam size (set 1 to run greedy search).
    num_hypotheses: int = 1,           # Number of hypotheses to return (should be <= beam_size
                                       # unless return_alternatives is set).
    length_penalty: float = 0,         # Length penalty constant to use during beam search.
    repetition_penalty: float = 1,     # Penalty applied to the score of previously generated tokens (set > 1 to penalize).
    disable_unk: bool = False,         # Disable the generation of the unknown token.
    allow_early_exit: bool = True,     # Allow the beam search to exit early when the first beam finishes.
    max_length: int = 512,             # Maximum generation length.
    min_length: int = 0,               # Minimum generation length.
    normalize_scores: bool = False,    # Normalize the score by the sequence length.
    return_scores: bool = False,       # Include the scores in the output.
    return_alternatives: bool = False, # Return alternatives at the first unconstrained decoding position.
    sampling_topk: int = 1,            # Randomly sample predictions from the top K candidates.
    sampling_temperature: float = 1,   # Sampling temperature to generate more random samples.
) -> Union[List[ctranslate2.GenerationResult], List[ctranslate2.AsyncGenerationResult]]
```

The result `ctranslate2.GenerationResult` has the following properties:

* `sequences`
* `scores` (empty if `return_scores` is set to `False`)

With `asynchronous=True`, the function returns a list of `ctranslate2.AsyncGenerationResult` instances. The actual `ctranslate2.GenerationResult` instance can be retrieved by calling `.result()` on the asynchronous wrapper.

Also see the [`GenerationOptions`](../include/ctranslate2/generation.h) structure for more details about the options.

### Batch scoring

Contrary to scoring with a translator, no special tokens are added to the input. If the model expects start or end tokens, the input should include these tokens.

```python
generator.score_batch(
    tokens: List[List[str]],
    *,
    max_batch_size: int = 0,       # Maximum batch size to run the model on.
    batch_type: str = "examples",  # Whether max_batch_size is the number of examples or tokens.
    max_input_length: int = 1024,  # Truncate inputs after this many tokens (set 0 to disable).
) -> List[List[float]]
```

## Utilities API

* `ctranslate2.__version__`<br/>Version of the Python package.
* `ctranslate2.contains_model(path: str)`<br/>Helper function to check if a directory seems to contain a CTranslate2 model.
* `ctranslate2.get_cuda_device_count()`<br/>Return the number of visible GPU devices.
* `ctranslate2.get_supported_compute_types(device: str, device_index: int = 0)`<br/>Return the set of supported compute types on a device.
* `ctranslate2.set_random_seed(seed: int)`<br/>Set the seed of random generators.

## Additional information

### Note on parallel execution

The `Translator` and `Generator` instances can be configured to process multiple batches in parallel:

```python
# Create a CPU translator with 4 workers each using 1 thread:
translator = ctranslate2.Translator(model_path, device="cpu", inter_threads=4, intra_threads=1)

# Create a GPU translator with 4 workers each running on a separate GPU:
translator = ctranslate2.Translator(model_path, device="cuda", device_index=[0, 1, 2, 3])

# Create a GPU translator with 4 workers each using a different CUDA stream:
translator = ctranslate2.Translator(model_path, device="cuda", inter_threads=4)
```

Parallel translations are enabled in the following cases:

* When calling `{translate,score}_file`.
* When calling `{translate,score,generate}_batch` and setting `max_batch_size`: the input will be split according to `max_batch_size` and each sub-batch will be translated in parallel.
* When calling `{translate,score,generate}_batch` from multiple Python threads: parallelization with Python threads is made possible because the `Translator` methods release the [Python GIL](https://wiki.python.org/moin/GlobalInterpreterLock).
* When calling `{translate,generate}_batch` multiple times with `asynchronous=True`:

```python
async_results = []
for batch in batch_generator():
    async_results.extend(translator.translate_batch(batch, asynchronous=True))

for async_result in async_results:
    print(async_result.result())  # This method blocks until the result is available.
```
