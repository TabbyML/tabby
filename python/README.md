# Python API

## Translator

```python
from ctranslate2 import translator

translator.initialize(4)  # Number of MKL threads.

t = translator.Translator(
    model_path: str      # Path to the CTranslate2 model directory.
    device="cpu",        # Can be "cpu", "cuda", or "auto".
    thread_pool_size=2)  # Number of concurrent translations.

output = t.translate_batch(
    tokens: list,            # A list of list of string.
    beam_size=4,             # Beam size.
    num_hypotheses=1,        # Number of hypotheses to return.
    length_penalty=0.6,      # Length penalty constant.
    max_decoding_steps=250,  # Maximum decoding steps.
    min_decoding_length=1,   # Minimum prediction length (EOS excluded).
    use_vmap=False)          # Use the VMAP saved in this model.

# output is a 2D list [batch x num_hypotheses] containing tuples of (score, tokens).

del t  # Release translator resources.
```

## Converter

Converters transform trained models to the CTranslate2 representation.

### Requirements

* Train a model that is compatible with one the model specifications supported by CTranslate2 (see the `--help` flag of the converters)
* Install the converter dependencies (e.g. TensorFlow or PyTorch)

### Examples

#### OpenNMT-tf

The OpenNMT-tf converter accepts both checkpoints and `SavedModel`. For example, let's convert a pretrained English-German Transformer model with INT16 quantization:

```bash
wget https://s3.amazonaws.com/opennmt-models/averaged-ende-export500k.tar.gz
tar xf averaged-ende-export500k.tar.gz

python -m ctranslate2.converters.opennmt_tf \
    --model_dir averaged-ende-export500k/1539080952/ \
    --output_dir ende_ctranslate2 \
    --model_spec TransformerBase --quantization int16
```

#### OpenNMT-py

The OpenNMT-py converter has a similar usage:

```bash
wget https://s3.amazonaws.com/opennmt-models/transformer-ende-wmt-pyOnmt.tar.gz
tar xf transformer-ende-wmt-pyOnmt.tar.gz

python -m ctranslate2.converters.opennmt_py \
    --model_path averaged-10-epoch.pt --output_dir ende_ctranslate2 \
    --model_spec TransformerBase --quantization int16
```

### Adding converters

Each converter should populate a model specification with trained weights coming from an existing model. The model specification declares the variable names and layout expected by the CTranslate2 core engine.

See the existing converters implementation which could be used as a template.
