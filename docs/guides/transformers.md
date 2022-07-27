# Transformers

CTranslate2 supports selected models from Hugging Face's [Transformers](https://github.com/huggingface/transformers). The following models are currently supported:

* BART
* M2M100
* MarianMT
* MBART
* OpenAI GPT2
* OPT
* Pegasus

The converter takes as argument the pretrained model name or the path to a model directory:

```bash
pip install transformers[torch]
ct2-transformers-converter --model facebook/m2m100_418M --output_dir ct2_model
```

## Special tokens in translation

For other frameworks, the `Translator` methods implicitly add special tokens to the source input when required. For example, models converted from Fairseq or Marian will implicitly append `</s>` to the source tokens.

However, these special tokens are not implicitly added for Transformers models since they are already returned by the corresponding tokenizer:

```python
>>> import transformers
>>> tokenizer = transformers.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
>>> tokenizer.convert_ids_to_tokens(tokenizer.encode("Hello world!"))
['▁Hello', '▁world', '!', '</s>']
```

```{important}
If you are not using the Hugging Face tokenizers, make sure to add these special tokens when required.
```

## BART

This example uses the [BART](https://huggingface.co/facebook/bart-large-cnn) model that was fine-tuned on CNN Daily Mail for text summarization.

```bash
ct2-transformers-converter --model facebook/bart-large-cnn --output_dir bart-large-cnn
```

```python
import ctranslate2
import transformers

translator = ctranslate2.Translator("bart-large-cnn")
tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

text = (
    "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    "amid dry conditions. "
    "The aim is to reduce the risk of wildfires. "
    "Nearly 800 thousand customers were scheduled to be affected by the shutoffs which "
    "were expected to last through at least midday tomorrow."
)

source = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
results = translator.translate_batch([source])
target = results[0].hypotheses[0]

print(tokenizer.decode(tokenizer.convert_tokens_to_ids(target), skip_special_tokens=True))
```

## MarianMT

This example uses the English-German model from [MarianMT](https://huggingface.co/docs/transformers/model_doc/marian).

```bash
ct2-transformers-converter --model Helsinki-NLP/opus-mt-en-de --output_dir opus-mt-en-de
```

```python
import ctranslate2
import transformers

translator = ctranslate2.Translator("opus-mt-en-de")
tokenizer = transformers.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")

source = tokenizer.convert_ids_to_tokens(tokenizer.encode("Hello world!"))
results = translator.translate_batch([source])
target = results[0].hypotheses[0]

print(tokenizer.decode(tokenizer.convert_tokens_to_ids(target)))
```

## M2M-100

This example uses the [M2M-100](https://huggingface.co/docs/transformers/model_doc/m2m_100) multilingual model.

```bash
ct2-transformers-converter --model facebook/m2m100_418M --output_dir m2m100_418
```

```python
import ctranslate2
import transformers

translator = ctranslate2.Translator("m2m100_418")
tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/m2m100_418M")
tokenizer.src_lang = "en"

source = tokenizer.convert_ids_to_tokens(tokenizer.encode("Hello world!"))
target_prefix = [tokenizer.lang_code_to_token["de"]]
results = translator.translate_batch([source], target_prefix=[target_prefix])
target = results[0].hypotheses[0][1:]

print(tokenizer.decode(tokenizer.convert_tokens_to_ids(target)))
```

## GPT-2

This example uses the small [GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2) model.

```bash
ct2-transformers-converter --model gpt2 --output_dir gpt2_ct2
```

```python
import ctranslate2
import transformers

generator = ctranslate2.Generator("gpt2_ct2")
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

# Unconditional generation.
start_tokens = [tokenizer.bos_token]
results = generator.generate_batch([start_tokens], max_length=30, sampling_topk=10)
output = results[0].sequences[0]
print(tokenizer.decode(tokenizer.convert_tokens_to_ids(output)))

# Conditional generation.
start_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode("It is"))
results = generator.generate_batch([start_tokens], max_length=30, sampling_topk=10)
output = results[0].sequences[0]
print(tokenizer.decode(tokenizer.convert_tokens_to_ids(output)))
```

## OPT

This example uses Meta's [OPT](https://huggingface.co/docs/transformers/model_doc/opt) model with 350M parameters. The usage is similar to GPT-2 but all inputs should start with the special token `</s>` which is automatically added by `GPT2Tokenizer`.

```{important}
Converting OPT models requires `transformers>=4.20.1`.
```

```bash
ct2-transformers-converter --model facebook/opt-350m --output_dir opt-350m-ct2
```

```python
import ctranslate2
import transformers

tokenizer = transformers.GPT2Tokenizer.from_pretrained("facebook/opt-350m")
generator = ctranslate2.Generator("opt-350m-ct2")

prompt = "Hey, are you conscious? Can you talk to me?"
start_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))

results = generator.generate_batch([start_tokens], max_length=30)
output_tokens = results[0].sequences[0]

output = tokenizer.decode(tokenizer.convert_tokens_to_ids(output_tokens))
print(output)
```
