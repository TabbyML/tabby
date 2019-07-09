# Python

## API

```python
from ctranslate2 import translator

t = translator.Translator(
    model_path: str          # Path to the CTranslate2 model directory.
    device="cpu",            # Can be "cpu", "cuda", or "auto".
    device_index=0,          # The index of the device to place this translator on.
    inter_threads=1,         # Maximum number of concurrent translations.
    intra_threads=4)         # Threads to use per translation.

# output is a 2D list [batch x num_hypotheses] containing dict with keys:
# * "score"
# * "tokens"
# * "attention" (if return_attention is set to True)
output = t.translate_batch(
    source: list,            # A list of list of string.
    target_prefix=None,      # An optional list of list of string.
    beam_size=4,             # Beam size.
    num_hypotheses=1,        # Number of hypotheses to return.
    length_penalty=0.6,      # Length penalty constant.
    max_decoding_length=250, # Maximum prediction length.
    min_decoding_length=1,   # Minimum prediction length.
    use_vmap=False,          # Use the VMAP saved in this model.
    return_attention=False)  # Also return the attention vectors.

t.translate_file(
    input_path: str,         # Input file.
    output_path: str,        # Output file.
    max_batch_size: int,     # Maximum batch size to translate.
    beam_size=4,             # Beam size.
    num_hypotheses=1,        # Number of hypotheses to output.
    length_penalty=0.6,      # Length penalty constant.
    max_decoding_length=250, # Maximum prediction length.
    min_decoding_length=1,   # Minimum prediction length.
    use_vmap=False,          # Use the VMAP saved in this model.
    with_scores=False)       # Also output predictions scores.

del t                        # Release the translator resources.
```
