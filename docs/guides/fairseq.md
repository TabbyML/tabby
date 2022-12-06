# Fairseq

CTranslate2 supports some Transformer models trained with [Fairseq](https://github.com/pytorch/fairseq/). The following model names are currently supported:

* `bart`
* `multilingual_transformer`
* `transformer`
* `transformer_align`
* `transformer_lm`

The conversion minimally requires the PyTorch model path and the Fairseq data directory which contains the vocabulary files:

```bash
pip install fairseq
ct2-fairseq-converter --model_path model.pt --data_dir data-bin/ --output_dir ct2_model
```

## Beam search equivalence

The default beam search parameters in CTranslate2 are different than Fairseq. Set the following parameters to match the Fairseq behavior:

```python
translator.translate_batch(tokens, beam_size=5)
```

## WMT16 English-German

Download and convert the pretrained [WMT16 English-German model](https://github.com/pytorch/fairseq/tree/main/examples/translation):

```bash
wget https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2
tar xf wmt16.en-de.joined-dict.transformer.tar.bz2

ct2-fairseq-converter --model_path wmt16.en-de.joined-dict.transformer/model.pt \
    --data_dir wmt16.en-de.joined-dict.transformer \
    --output_dir ende_ctranslate2
```

The converted model can then be used on tokenized inputs:

```python
import ctranslate2

translator = ctranslate2.Translator("ende_ctranslate2/", device="cpu")
results = translator.translate_batch([["H@@", "ello", "world@@", "!"]])

print(results[0].hypotheses[0])
```

```{note}
For simplicity, this example does not show how to tokenize the text. The tokens are obtained by running `sacremoses` and applying the BPE codes included in the model.
```

## WMT19 language model

The FAIR team published [pretrained language models](https://github.com/pytorch/fairseq/blob/main/examples/language_model/README.md) as part of the WMT19 news translation task. They can be converted to the CTranslate2 format:

```bash
wget https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.en.tar.gz
tar xf wmt19.en.tar.gz

ct2-fairseq-converter --data_dir wmt19.en/ --model_path wmt19.en/model.pt --output_dir wmt19_en_ct2
```

The model can then be used to sample or score sequences of tokens. All inputs should start with the special token `</s>`:

```python
import numpy as np
import ctranslate2

generator = ctranslate2.Generator("wmt19_en_ct2/", device="cpu")

# Sample from the language model.
results = generator.generate_batch([["</s>", "The"]], sampling_topk=10, max_length=50)
print(results[0].sequences[0])

# Compute the perplexity for a sentence.
outputs = generator.score_batch([["</s>", "The", "sky", "is", "blue", "."]])
perplexity = np.exp(-np.mean(outputs[0].log_probs))
print(perplexity)
```

```{note}
For simplicity, this example does not show how to tokenize the text. The tokens are obtained by running `sacremoses` and applying the BPE codes included in the model.
```

## M2M-100

The pretrained multilingual model [M2M-100](https://github.com/pytorch/fairseq/tree/main/examples/m2m_100) can also be used in CTranslate2. The conversion option `--fixed_dictionary` is required for this model that uses a single vocabulary file:

```bash
# 418M parameters:
wget https://dl.fbaipublicfiles.com/m2m_100/418M_last_checkpoint.pt

# 1.2B parameters:
wget https://dl.fbaipublicfiles.com/m2m_100/1.2B_last_checkpoint.pt

wget https://dl.fbaipublicfiles.com/m2m_100/model_dict.128k.txt
wget https://dl.fbaipublicfiles.com/m2m_100/spm.128k.model

ct2-fairseq-converter --data_dir . --model_path 418M_last_checkpoint.pt \
    --fixed_dictionary model_dict.128k.txt \
    --output_dir m2m_100_418m_ct2
```

For translation, the language tokens should prefix the source and target sequences. Language tokens have the format `__X__` where `X` is the language code. See the end of the fixed dictionary file for the list of accepted languages.

```python
import ctranslate2
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("spm.128k.model")

source = ["__en__"] + sp.encode("Hello world!", out_type=str)
target_prefix = ["__de__"]

translator = ctranslate2.Translator("m2m_100_418m_ct2")
result = translator.translate_batch([source], target_prefix=[target_prefix])

output = sp.decode(result[0].hypotheses[0][1:])
print(output)
```

## MBART-50

[MBART-50](https://github.com/pytorch/fairseq/blob/main/examples/multilingual/README.md) is another pretrained multilingual translation model.

```bash
wget https://dl.fbaipublicfiles.com/fairseq/models/mbart50/mbart50.ft.nn.tar.gz
tar xf mbart50.ft.nn.tar.gz

ct2-fairseq-converter --data_dir mbart50.ft.nn/ --model_path mbart50.ft.nn/model.pt \
    --output_dir mbart50_ct2
```

Similar to M2M-100, the language tokens should prefix the source and target sequences. The list of language tokens is defined in the file `mbart50.ft.nn/ML50_langs.txt`.

```python
import ctranslate2
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("mbart50.ft.nn/sentence.bpe.model")

source = sp.encode("UN Chief Says There Is No Military Solution in Syria", out_type=str)
source = ["[en_XX]"] + source
target_prefix = ["[ro_RO]"]

translator = ctranslate2.Translator("mbart50_ct2")
result = translator.translate_batch([source], target_prefix=[target_prefix])

output = sp.decode(result[0].hypotheses[0][1:])
print(output)
```
