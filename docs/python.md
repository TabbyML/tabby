# Python

```python
import ctranslate2
```

## Model conversion API

```python
converter = ctranslate2.converters.OpenNMTTFConverter(
    model_path: str = None,  # Path to a OpenNMT-tf checkpoint or SavedModel (mutually exclusive with variables)
    src_vocab: str = None,   # Path to the source vocabulary (required for checkpoints).
    tgt_vocab: str = None,   # Path to the target vocabulary (required for checkpoints).
    variables: dict = None,  # Dict of variables name to value (mutually exclusive with model_path).
)

converter = ctranslate2.converters.OpenNMTPyConverter(
    model_path: str,         # Path to the OpenNMT-py model.
)

output_dir = converter.convert(
    output_dir: str,          # Path to the output directory.
    model_spec: ModelSpec,    # A model specification instance from ctranslate2.specs.
    vmap: str = None,         # Path to a vocabulary mapping file.
    quantization: str = None, # Weights quantization: "int8" or "int16".
    force: bool = False,      # Override output_dir if it exists.
)
```

## Translation API

```python
translator = ctranslate2.Translator(
    model_path: str                 # Path to the CTranslate2 model directory.
    device: str = "cpu",            # The device to use: "cpu", "cuda", or "auto".

    # The device ID, or list of device IDs, where to place this translator on.
    device_index: Union[int, List[int]] = 0,

    # The computation type: "default", "auto", "int8", "int16", "float16", or "float",
    # or a dict mapping a device to a computation type.
    compute_type: Union[str, Dict[str, str]] = "default",

    inter_threads: int = 1,         # Maximum number of parallel translations (CPU only).
    intra_threads: int = 4,         # Threads to use per translation (CPU only).
)

# Properties:
translator.device              # Device this translator is running on.
translator.device_index        # List of device IDs where this translator is running on.
translator.num_translators     # Number of translators backing this instance.
translator.num_queued_batches  # Number of batches waiting to be translated.

# output is a 2D list [batch x num_hypotheses] containing dict with keys:
# * "tokens"
# * "score" (if return_scores is set to True)
# * "attention" (if return_attention is set to True)
output = translator.translate_batch(
    source: list,                      # A list of list of string.
    target_prefix: list = None,        # An optional list of list of string.
    max_batch_size: int = 0,           # Maximum batch size to run the model on.
    batch_type: str = "examples",      # Whether max_batch_size is the number of examples or tokens.
    beam_size: int = 2,                # Beam size (set 1 to run greedy search).
    num_hypotheses: int = 1,           # Number of hypotheses to return (should be <= beam_size
                                       # unless return_alternatives is set).
    length_penalty: float = 0,         # Length penalty constant to use during beam search.
    coverage_penalty: float = 0,       # Converage penalty constant to use during beam search.
    max_decoding_length: int = 250,    # Maximum prediction length.
    min_decoding_length: int = 1,      # Minimum prediction length.
    use_vmap: bool = False,            # Use the vocabulary mapping file saved in this model.
    return_scores: bool = True,        # Include the prediction scores in the output.
    return_attention: bool = False,    # Include the attention vectors in the output.
    return_alternatives: bool = False, # Return alternatives at the first unconstrained decoding position.
    sampling_topk: int = 1,            # Randomly sample predictions from the top K candidates (with beam_size=1).
    sampling_temperature: float = 1,   # Sampling temperature to generate more random samples.
    replace_unknowns: bool = False,    # Replace unknown target tokens by the source token with the highest attention.
)

# stats is a tuple of file statistics containing in order:
# 1. the number of generated target tokens
# 2. the number of translated examples
# 3. the total translation time in milliseconds
stats = translator.translate_file(
    input_path: str,                # Input file.
    output_path: str,               # Output file.
    max_batch_size: int = 32,       # Maximum batch size to run the model on.
    read_batch_size: int = 0,       # Number of sentences to read at once.
    batch_type: str = "examples",   # Whether the batch size is the number of examples or tokens.
    beam_size: int = 2,
    num_hypotheses: int = 1,
    length_penalty: float = 0,
    coverage_penalty: float = 0,
    max_decoding_length: int = 250,
    min_decoding_length: int = 1,
    use_vmap: bool = False,
    with_scores: bool = False,
    sampling_topk: int = 1,
    sampling_temperature: float = 1,
    tokenize_fn: callable = None,   # Function with signature: string -> list of strings
    detokenize_fn: callable = None, # Function with signature: list of strings -> string
    target_path: str = "",          # Target prefix file.
    target_tokenize_fn: callable = None,  # Same as tokenize_fn but for the target.
    replace_unknowns: bool = False,  # Replace unknown target tokens by the source token with the highest attention.
)
```

Also see the [`TranslationOptions`](../include/ctranslate2/translator.h) structure for more details about the options.

### Note on parallel translations

A `Translator` instance can be configured to process multiple batches in parallel:

```python
# Create a CPU translator with 4 workers each using 1 thread:
translator = ctranslate2.Translator(model_path, device="cpu", inter_threads=4, intra_threads=1)

# Create a GPU translator with 4 workers each running on a separate GPU:
translator = ctranslate2.Translator(model_path, device="cuda", device_index=[0, 1, 2, 3])
```

Parallel translations are enabled in the following cases:

* When calling `translate_file`.
* When calling `translate_batch` and setting `max_batch_size`: the input will be split according to `max_batch_size` and each sub-batch will be translated in parallel.
* When calling `translate_batch` from multiple Python threads. If you are using a multithreaded HTTP server, this may already be the case. For other cases, you could use a [`ThreadPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor) to submit multiple concurrent translations:

```python
import concurrent.futures

with concurrent.futures.ThreadPoolExecutor(max_workers=inter_threads) as executor:
    futures = [
        executor.submit(translator.translate_batch, batch)
        for batch in batch_generator()
    ]
    for future in futures:
        translation_result = future.result()
```

Note: parallelization with Python threads is made possible because the `Translator` methods release the [Python GIL](https://wiki.python.org/moin/GlobalInterpreterLock).

## Memory management API

* `translator.unload_model(to_cpu: bool = False)`<br/>Unload the model attached to this translator but keep enough runtime context to quickly resume translation on the initial device. When `to_cpu` is `True`, the model is moved to the CPU memory and not fully unloaded.
* `translator.load_model()`<br/>Load the model back to the initial device.
* `translator.model_is_loaded`<br/>Property set to `True` when the model is loaded on the initial device and ready to be used.
* `del translator`<br/>Release the translator resources.

When using multiple Python threads, the application should ensure that no translations are running before calling these functions.

## Utility API

* `ctranslate2.contains_model(path: str)`<br/>Helper function to check if a directory seems to contain a CTranslate2 model.
