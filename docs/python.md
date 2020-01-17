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
    model_spec: LayerSpec,   # A model specification instance from ctranslate2.specs.
    vmap=None,               # Path to a vocabulary mapping file.
    quantization=None,       # Weights quantization, can be "int8" or "int16".
    force=False)             # Override output_dir if it exists.
```

## Translation API

```python
import ctranslate2

translator = ctranslate2.Translator(
    model_path: str          # Path to the CTranslate2 model directory.
    device="cpu",            # Can be "cpu", "cuda", or "auto".
    device_index=0,          # The index of the device to place this translator on.
    compute_type="default"   # The final data type to convert. Can be "default", "int8", "int16" and "float"
    inter_threads=1,         # Maximum number of concurrent translations.
    intra_threads=4)         # Threads to use per translation.

# output is a 2D list [batch x num_hypotheses] containing dict with keys:
# * "score"
# * "tokens"
# * "attention" (if return_attention is set to True)
output = translator.translate_batch(
    source: list,            # A list of list of string.
    target_prefix=None,      # An optional list of list of string.
    beam_size=4,             # Beam size.
    num_hypotheses=1,        # Number of hypotheses to return.
    length_penalty=0,        # Length penalty constant.
    max_decoding_length=250, # Maximum prediction length.
    min_decoding_length=1,   # Minimum prediction length.
    use_vmap=False,          # Use the VMAP saved in this model.
    return_attention=False,  # Also return the attention vectors.
    sampling_topk=1,         # Randomly sample from the top K candidates.
    sampling_temperature=1.) # Sampling temperature.

translator.translate_file(
    input_path: str,         # Input file.
    output_path: str,        # Output file.
    max_batch_size: int,     # Maximum batch size to translate.
    beam_size=4,             # Beam size.
    num_hypotheses=1,        # Number of hypotheses to output.
    length_penalty=0,        # Length penalty constant.
    max_decoding_length=250, # Maximum prediction length.
    min_decoding_length=1,   # Minimum prediction length.
    use_vmap=False,          # Use the VMAP saved in this model.
    with_scores=False,       # Also output predictions scores.
    sampling_topk=1,         # Randomly sample from the top K candidates.
    sampling_temperature=1.) # Sampling temperature.

del translator               # Release the translator resources.
```
