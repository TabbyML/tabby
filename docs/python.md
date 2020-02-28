# Python

## Model conversion API

```python
import ctranslate2

converter = ctranslate2.converters.OpenNMTTFConverter(
    model_path=None, # Path to a OpenNMT-tf checkpoint or SavedModel (mutually exclusive with variables)
    src_vocab=None,  # Path to the source vocabulary (required for checkpoints).
    tgt_vocab=None,  # Path to the target vocabulary (required for checkpoints).
    variables=None)  # Dict of variables name to value (mutually exclusive with model_path).

converter = ctranslate2.converters.OpenNMTPyConverter(
    model_path: str)         # Path to the OpenNMT-py model.

output_dir = converter.convert(
    output_dir: str,         # Path to the output directory.
    model_spec: ModelSpec,   # A model specification instance from ctranslate2.specs.
    vmap=None,               # Path to a vocabulary mapping file.
    quantization=None,       # Weights quantization: "int8" or "int16".
    force=False)             # Override output_dir if it exists.
```

## Translation API

```python
import ctranslate2

translator = ctranslate2.Translator(
    model_path: str          # Path to the CTranslate2 model directory.
    device="cpu",            # The device to use: "cpu", "cuda", or "auto".
    device_index=0,          # The index of the device to place this translator on.
    compute_type="default"   # The computation type: "default", "int8", "int16", or "float".
    inter_threads=1,         # Maximum number of concurrent translations (CPU only).
    intra_threads=4)         # Threads to use per translation (CPU only).

# output is a 2D list [batch x num_hypotheses] containing dict with keys:
# * "score"
# * "tokens"
# * "attention" (if return_attention is set to True)
output = translator.translate_batch(
    source: list,              # A list of list of string.
    target_prefix=None,        # An optional list of list of string.
    max_batch_size=0,          # Maximum batch size to run the model on.
    beam_size=2,               # Beam size (set 1 to run greedy search).
    num_hypotheses=1,          # Number of hypotheses to return (should be <= beam_size).
    length_penalty=0,          # Length penalty constant to use during beam search.
    max_decoding_length=250,   # Maximum prediction length.
    min_decoding_length=1,     # Minimum prediction length.
    use_vmap=False,            # Use the vocabulary mapping file saved in this model.
    return_attention=False,    # Include the attention vectors in the output.
    return_alternatives=False, # Return alternatives at the first unconstrained decoding position.
    sampling_topk=1,           # Randomly sample predictions from the top K candidates (with beam_size=1).
    sampling_temperature=1.)   # Sampling temperature to generate more random samples.

# stats is a tuple of file statistics containing in order:
# 1. the number of generated target tokens
stats = translator.translate_file(
    input_path: str,         # Input file.
    output_path: str,        # Output file.
    max_batch_size: int,     # Maximum batch size to run the model on.
    read_batch_size=0,       # Number of sentences to read at once.
    beam_size=2,
    num_hypotheses=1,
    length_penalty=0,
    max_decoding_length=250,
    min_decoding_length=1,
    use_vmap=False,
    with_scores=False,
    sampling_topk=1,
    sampling_temperature=1.)

del translator               # Release the translator resources.
```

Also see the [`TranslationOptions`](../include/ctranslate2/translator.h) structure for more details about the options.
