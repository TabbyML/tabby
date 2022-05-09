# Quickstart

Start using CTranslate2 by converting a pretrained model and running your first translation.

**1\. Install the Python packages**

```bash
pip install ctranslate2 OpenNMT-py sentencepiece
```

**2\. Download the English-German Transformer model trained with OpenNMT-py**

```bash
wget https://s3.amazonaws.com/opennmt-models/transformer-ende-wmt-pyOnmt.tar.gz
tar xf transformer-ende-wmt-pyOnmt.tar.gz
```

**3\. Convert the model to the CTranslate2 format**

```bash
ct2-opennmt-py-converter --model_path averaged-10-epoch.pt --output_dir ende_ctranslate2
```

**4\. Translate texts with the Python API**

```python
import ctranslate2
import sentencepiece as spm

translator = ctranslate2.Translator("ende_ctranslate2/", device="cpu")
sp = spm.SentencePieceProcessor("sentencepiece.model")

input_text = "Hello world!"
input_tokens = sp.encode(input_text, out_type=str)

results = translator.translate_batch([input_tokens])

output_tokens = results[0].hypotheses[0]
output_text = sp.decode(output_tokens)

print(output_text)
```

This code should print the sentence:

> Hallo Welt!

If that's the case, you successfully converted and executed a translation model with CTranslate2! Consider browsing the other sections for more information and examples.
