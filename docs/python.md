# Python

```python
import ctranslate2
```

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
)

output_dir = converter.convert(
    output_dir: str,          # Path to the output directory.
    vmap: str = None,         # Path to a vocabulary mapping file.
    quantization: str = None, # Weights quantization: "int8", "int8_float16", "int16", or "float16".
    force: bool = False,      # Override output_dir if it exists.
)
```

## Translation API

```python
translator = ctranslate2.Translator(
    model_path: str                 # Path to the CTranslate2 model directory.
    device: str = "cpu",            # The device to use: "cpu", "cuda", or "auto".
    *,

    # The device ID, or list of device IDs, where to place this translator on.
    device_index: Union[int, List[int]] = 0,

    # The computation type: "default", "auto", "int8", "int8_float16", "int16", "float16", or "float",
    # or a dict mapping a device to a computation type.
    compute_type: Union[str, Dict[str, str]] = "default",

    inter_threads: int = 1,         # Maximum number of parallel translations (CPU only).
    intra_threads: int = 0,         # Threads to use per translation (CPU only).
                                    # Set 0 to use a default value.
)

# Properties:
translator.device              # Device this translator is running on.
translator.device_index        # List of device IDs where this translator is running on.
translator.num_translators     # Number of translators backing this instance.
translator.num_queued_batches  # Number of batches waiting to be translated.

# results is a list of TranslationResult instances that have the following properties:
# * hypotheses
# * scores (empty if return_scores is set to False)
# * attention (empty if return_attention is set to False)
# With asynchronous=True, the function returns a list of AsyncTranslationResult instances.
# The actual TranslationResult instance can be retrieved by calling .result() on the async wrapper.
results = translator.translate_batch(
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
    prefix_bias_beta: float = 0,       # Parameter for biasing translations towards given prefix.
    allow_early_exit: bool = True,     # Allow the beam search to exit early when the first beam finishes.
    max_decoding_length: int = 250,    # Maximum prediction length.
    min_decoding_length: int = 1,      # Minimum prediction length.
    use_vmap: bool = False,            # Use the vocabulary mapping file saved in this model.
    normalize_scores: bool = False,    # Normalize the score by the hypothesis length.
    return_scores: bool = False,       # Include the prediction scores in the output.
    return_attention: bool = False,    # Include the attention vectors in the output.
    return_alternatives: bool = False, # Return alternatives at the first unconstrained decoding position.
    sampling_topk: int = 1,            # Randomly sample predictions from the top K candidates (with beam_size=1).
    sampling_temperature: float = 1,   # Sampling temperature to generate more random samples.
    replace_unknowns: bool = False,    # Replace unknown target tokens by the source token with the highest attention.
)

# stats is a TranslationStats instance that has the following properties:
# * num_tokens: the number of generated target tokens
# * num_examples: the number of translated examples
# * total_time_in_ms: the total translation time in milliseconds
stats = translator.translate_file(
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
    prefix_bias_beta: float = 0,
    allow_early_exit: bool = True,
    max_decoding_length: int = 250,
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
)
```

Also see the [`TranslationOptions`](../include/ctranslate2/translator.h) structure for more details about the options.

## Scoring API

The `Translator` object (see previous section) can also be used to score existing translations:

```python
# Batch scoring:
scores = translator.score_batch(
    source: List[List[str]],
    target: List[List[str]],
    *,
    max_batch_size: int = 0,       # Maximum batch size to run the model on.
    batch_type: str = "examples",  # Whether max_batch_size is the number of examples or tokens.
)

# File scoring:
# Each line in output_path will have the format: <score> ||| <target>
# The score is normalized by the target length.
#
# The returned stats object has the following properties:
# * num_tokens: the number of scored target tokens
# * num_examples: the number of scored examples
# * total_time_in_ms: the total scoring time in milliseconds
stats = translator.score_file(
    source_path: str,              # Source file.
    target_path: str,              # Target file.
    output_path: str,              # Output file.
    *,
    max_batch_size: int = 32,      # Maximum batch size to run the model on.
    read_batch_size: int = 0,      # Number of sentences to read at once.
    batch_type: str = "examples",  # Whether the batch size is the number of examples or tokens.
    source_tokenize_fn: callable = None,   # Function with signature: string -> list of strings
    target_tokenize_fn: callable = None,   # Function with signature: string -> list of strings
    target_detokenize_fn: callable = None, # Function with signature: list of strings -> string
)
```

### Note on parallel execution

A `Translator` instance can be configured to process multiple batches in parallel:

```python
# Create a CPU translator with 4 workers each using 1 thread:
translator = ctranslate2.Translator(model_path, device="cpu", inter_threads=4, intra_threads=1)

# Create a GPU translator with 4 workers each running on a separate GPU:
translator = ctranslate2.Translator(model_path, device="cuda", device_index=[0, 1, 2, 3])
```

Parallel translations are enabled in the following cases:

* When calling `translate_file` (or `score_file`).
* When calling `translate_batch` (or `score_batch`) and setting `max_batch_size`: the input will be split according to `max_batch_size` and each sub-batch will be translated in parallel.
* When calling `translate_batch` (or `score_batch`) from multiple Python threads: parallelization with Python threads is made possible because the `Translator` methods release the [Python GIL](https://wiki.python.org/moin/GlobalInterpreterLock).
* When calling `translate_batch` multiple times with `asynchronous=True`:

```python
async_results = []
for batch in batch_generator():
    async_results.extend(translator.translate_batch(batch, asynchronous=True))

for async_result in async_results:
    print(async_result.result())  # This method blocks until the result is available.
```

## Memory management API

* `translator.unload_model(to_cpu: bool = False)`<br/>Unload the model attached to this translator but keep enough runtime context to quickly resume translation on the initial device. The model is not guaranteed to be unloaded if the translator is used simultaneously in another thread.
  * `to_cpu`: If `True`, the model is moved to the CPU memory and not fully unloaded.
* `translator.load_model()`<br/>Load the model back to the initial device.
* `translator.model_is_loaded`<br/>Property set to `True` when the model is loaded on the initial device and ready to be used.
* `del translator`<br/>Release the translator resources.

## Utility API

* `ctranslate2.__version__`<br/>Version of the Python package.
* `ctranslate2.contains_model(path: str)`<br/>Helper function to check if a directory seems to contain a CTranslate2 model.
* `ctranslate2.get_cuda_device_count()`<br/>Return the number of visible GPU devices.
* `ctranslate2.get_supported_compute_types(device: str, device_index: int = 0)`<br/>Return the set of supported compute types on a device.
