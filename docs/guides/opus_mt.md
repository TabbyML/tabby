# OPUS-MT

[OPUS-MT](https://github.com/Helsinki-NLP/OPUS-MT) contains a collection of 1000+ pretrained models. Since the models are Transformer models trained with [Marian](https://github.com/marian-nmt/marian), they are compatible with CTranslate2.

The [Marian guide](marian.md) also applies for these models, but a separate converter is provided for convenience:

```bash
ct2-opus-mt-converter --model_dir opus_model --output_dir ct2_model
```

## Example

This example uses the English-German model:

```
wget https://object.pouta.csc.fi/OPUS-MT-models/en-de/opus-2020-02-26.zip
unzip opus-2020-02-26.zip

ct2-opus-mt-converter --model_dir . --output_dir ende_ctranslate2
```

```python
import ctranslate2
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("source.spm")

source = sp.encode("Hello world!", out_type=str)

translator = ctranslate2.Translator("ende_ctranslate2")
results = translator.translate_batch([source])

output = sp.decode(results[0].hypotheses[0])
print(output)
```
