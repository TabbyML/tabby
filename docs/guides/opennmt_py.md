# OpenNMT-py

CTranslate2 supports Transformer models trained with [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py). The conversion simply requires the PyTorch model path, e.g.:

```bash
pip install OpenNMT-py==2.*
ct2-opennmt-py-converter --model_path model.pt --output_dir ct2_model
```

Alternatively, you can also convert the model directly from OpenNMT-py with the model release script:

```bash
onmt_release_model --model model.pt --format ctranslate2 --output ct2_model
```

```{tip}
See the [quickstart](../quickstart.md) for a complete example using an OpenNMT-py model.
```

## Text generation with `transformer_lm`

Decoder-only models using the `transformer_lm` decoder type are supported and can be converted with the same command line.

During generation, make sure to always include `<s>` in the start tokens, e.g.:

```python
generator = ctranslate2.Generator(model_path)
generator.generate_batch([["<s>", "▁Hello"]])
```
