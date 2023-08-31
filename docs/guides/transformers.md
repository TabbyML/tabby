# Transformers

CTranslate2 supports selected models from Hugging Face's [Transformers](https://github.com/huggingface/transformers). The following models are currently supported:

* BART
* BERT
* BLOOM
* CodeGen
* DistilBERT
* Falcon
* Llama
* M2M100
* MarianMT
* MBART
* MPT
* NLLB
* OpenAI GPT2
* GPTBigCode
* GPT-J
* GPT-NeoX
* OPT
* Pegasus
* T5
* Whisper
* XLM-RoBERTa

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

## BERT

[BERT](https://huggingface.co/docs/transformers/model_doc/bert) is pretrained model on English language using a masked language modeling objective.

CTranslate2 only implements the `BertModel` class from Transformers which includes the Transformer encoder and the pooling layer. Task-specific layers should be run with PyTorch as shown in the example below.

```bash
ct2-transformers-converter --model textattack/bert-base-uncased-yelp-polarity --output_dir bert-base-uncased-yelp-polarity
```

```python
import ctranslate2
import numpy as np
import torch
import transformers

device = "cuda"
encoder = ctranslate2.Encoder("bert-base-uncased-yelp-polarity", device=device)

model_name = "textattack/bert-base-uncased-yelp-polarity"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
classifier = transformers.AutoModelForSequenceClassification.from_pretrained(model_name).classifier
classifier.eval()
classifier.to(device)

inputs = ["It was good!", "Worst experience in my life.", "It was not good."]
tokens = tokenizer(inputs).input_ids

output = encoder.forward_batch(tokens)
pooler_output = output.pooler_output

if device == "cuda":
    pooler_output = torch.as_tensor(pooler_output, device=device)
else:
    pooler_output = np.array(pooler_output)
    pooler_output = torch.as_tensor(pooler_output)

logits = classifier(pooler_output)
predicted_class_ids = logits.argmax(1)

print(predicted_class_ids)
```

## BLOOM

[BLOOM](https://huggingface.co/docs/transformers/model_doc/bloom) is a collection of multilingual language models trained by the [BigScience workshop](https://bigscience.huggingface.co/).

This example uses the smallest model with 560M parameters.

```bash
ct2-transformers-converter --model bigscience/bloom-560m --output_dir bloom-560m
```

```python
import ctranslate2
import transformers

generator = ctranslate2.Generator("bloom-560m")
tokenizer = transformers.AutoTokenizer.from_pretrained("bigscience/bloom-560m")

text = "Hello, I am"
start_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
results = generator.generate_batch([start_tokens], max_length=30, sampling_topk=10)
print(tokenizer.decode(results[0].sequences_ids[0]))
```

## DistilBERT

[DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert) is a small, fast, cheap and light Transformer Encoder model trained by distilling BERT base.

CTranslate2 only implements the `DistilBertModel` class from Transformers which includes the Transformer encoder. Task-specific layers should be run with PyTorch, similar to the example for {ref}`guides/transformers:bert`.

```bash
ct2-transformers-converter --model distilbert-base-uncased --output_dir distilbert-base-uncased
```

## Falcon

[Falcon](https://huggingface.co/tiiuae/falcon-7b) is a collection of generative language models trained by [TII](https://www.tii.ae/). The example below uses "Falcon-7B-Instruct" which is based on "Falcon-7B" and finetuned on a mixture of chat/instruct datasets.

```bash
ct2-transformers-converter --model tiiuae/falcon-7b-instruct --quantization float16 --output_dir falcon-7b-instruct --trust_remote_code
```

```python
import ctranslate2
import transformers

generator = ctranslate2.Generator("falcon-7b-instruct", device="cuda")
tokenizer = transformers.AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")

prompt = (
    "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. "
    "Giraftron believes all other animals are irrelevant when compared to the glorious majesty "
    "of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:"
)

tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))

results = generator.generate_batch([tokens], sampling_topk=10, max_length=200, include_prompt_in_result=False)
output = tokenizer.decode(results[0].sequences_ids[0])

print(output)
```

## Llama 2

[Llama 2](https://ai.meta.com/llama/) is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters.

The models with the suffix "-hf" such as [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) can be converted with the Transformers converter. For example:

```bash
ct2-transformers-converter --model meta-llama/Llama-2-7b-chat-hf --quantization float16 --output_dir llama-2-7b-chat-ct2
```

```{important}
You need to request an access to the Llama 2 models before you can download them from the Hugging Face Hub. See the instructions on the [model page](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf). Once you have access to the model, you should login with `huggingface-cli login` before running the conversion command.
```

```{seealso}
The example [Chat with Llama 2](https://github.com/OpenNMT/CTranslate2/tree/master/examples/llama2) which demonstrates how to implement an interactive chat session using CTranslate2.
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

## MPT

[MPT-7B](https://huggingface.co/mosaicml/mpt-7b) is a decoder-style transformer pretrained from scratch on 1T tokens of English text and code. This model was trained by [MosaicML](https://www.mosaicml.com/).

```{note}
The code is included in the model so you should pass `--trust_remote_code` to the conversion command.
```

```bash
ct2-transformers-converter --model mosaicml/mpt-7b --output_dir mpt-7b --quantization int8_float16 --trust_remote_code
```

```python
import ctranslate2
import transformers

generator = ctranslate2.Generator("mpt-7b")
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

prompt = "In a shocking finding, scientists"
tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))

results = generator.generate_batch([tokens], max_length=30, sampling_topk=10)

text = tokenizer.decode(results[0].sequences_ids[0])
print(text)
```

## NLLB

[NLLB](https://huggingface.co/docs/transformers/model_doc/nllb) is a collection of multilingual models trained by Meta and supporting 200 languages. See [here](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200) for the list of accepted language codes.

The example below uses the smallest version with 600M parameters.

```{important}
Converting NLLB models requires `transformers>=4.21.0`.
```

```bash
ct2-transformers-converter --model facebook/nllb-200-distilled-600M --output_dir nllb-200-distilled-600M
```

```python
import ctranslate2
import transformers

src_lang = "eng_Latn"
tgt_lang = "fra_Latn"

translator = ctranslate2.Translator("nllb-200-distilled-600M")
tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=src_lang)

source = tokenizer.convert_ids_to_tokens(tokenizer.encode("Hello world!"))
target_prefix = [tgt_lang]
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
print(tokenizer.decode(results[0].sequences_ids[0]))

# Conditional generation.
start_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode("It is"))
results = generator.generate_batch([start_tokens], max_length=30, sampling_topk=10)
print(tokenizer.decode(results[0].sequences_ids[0]))
```

## GPTBigCode

[GPTBigCode](https://huggingface.co/docs/transformers/model_doc/gpt_bigcode) model was first proposed in SantaCoder: don’t reach for the stars, and used by models like StarCoder.

```bash
ct2-transformers-converter --model bigcode/starcoder --revision main --quantization float16 --output_dir starcoder_ct2
```

```python
import ctranslate2
import transformers

generator = ctranslate2.Generator("starcoder_ct2")
tokenizer = transformers.AutoTokenizer.from_pretrained("bigcode/starcoder")

prompt = "<fim_prefix>def print_hello_world():\n    <fim_suffix>\n    print('Hello world!')<fim_middle>"
tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))

results = generator.generate_batch([tokens], max_length=30, include_prompt_in_result=False)

text = tokenizer.decode(results[0].sequences_ids[0])
print(text)
```

## GPT-J

[GPT-J](https://huggingface.co/docs/transformers/model_doc/gptj) is a GPT-2-like language model trained on the Pile dataset. The example below uses the version with 6B parameters:

```bash
ct2-transformers-converter --model EleutherAI/gpt-j-6B --revision float16 --quantization float16 --output_dir gptj_ct2
```

```{note}
To reduce the memory usage during conversion, the command above uses the [float16 branch](https://huggingface.co/EleutherAI/gpt-j-6b/tree/float16) of the model and saves the weights in FP16. Still, the conversion will use up to 24GB of memory.
```

```python
import ctranslate2
import transformers

generator = ctranslate2.Generator("gptj_ct2")
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

prompt = "In a shocking finding, scientists"
tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))

results = generator.generate_batch([tokens], max_length=30, sampling_topk=10)

text = tokenizer.decode(results[0].sequences_ids[0])
print(text)
```

## GPT-NeoX

The [GPT-NeoX](https://huggingface.co/docs/transformers/model_doc/gpt_neox) architecture was first introduced by EleutherAI with "GPT-NeoX-20B", a 20 billion parameter autoregressive language model trained on the Pile.

```bash
ct2-transformers-converter --model EleutherAI/gpt-neox-20b --quantization float16 --output_dir gpt_neox_ct2
```

```python
import ctranslate2
import transformers

generator = ctranslate2.Generator("gpt_neox_ct2")
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

prompt = "GPTNeoX20B is a 20B-parameter autoregressive Transformer model developed by EleutherAI."
tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))

results = generator.generate_batch(
    [tokens],
    max_length=64,
    sampling_topk=20,
    sampling_temperature=0.9,
)

text = tokenizer.decode(results[0].sequences_ids[0])
print(text)
```

## OPT

This example uses Meta's [OPT](https://huggingface.co/docs/transformers/model_doc/opt) model with 350M parameters. The usage is similar to GPT-2 but all inputs should start with the special token `</s>` which is automatically added by `GPT2Tokenizer`.

```{important}
Converting OPT models requires `transformers>=4.20.1`.
```

```{tip}
If you plan to [quantize](../quantization.md) OPT models to 8-bit, it is recommended to download the corresponding activation scales from the [SmoothQuant repository](https://github.com/mit-han-lab/smoothquant/tree/main/act_scales) and pass them to the converter option `--activation_scales`. Some weights will be rescaled to smooth the intermediate activations and improve the quantization accuracy.
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

output = tokenizer.decode(results[0].sequences_ids[0])
print(output)
```

## T5

[T5](https://huggingface.co/docs/transformers/model_doc/t5) is an encoder-decoder model pre-trained on a multi-task mixture of unsupervised and supervised tasks and for which each task is converted into a text-to-text format.

The example below uses the `t5-small` and machine translation input.

```{note}
The variants T5v1.1, mT5, and FLAN-T5 are also supported.
```

```bash
ct2-transformers-converter --model t5-small --output_dir t5-small-ct2
```

```python
import ctranslate2
import transformers

translator = ctranslate2.Translator("t5-small-ct2")
tokenizer = transformers.AutoTokenizer.from_pretrained("t5-small")

input_text = "translate English to German: The house is wonderful."
input_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(input_text))

results = translator.translate_batch([input_tokens])

output_tokens = results[0].hypotheses[0]
output_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(output_tokens))

print(output_text)
```

## Whisper

[Whisper](https://huggingface.co/docs/transformers/model_doc/whisper) is a multilingual speech recognition model published by OpenAI.

```{important}
Converting Whisper models requires `transformers>=4.23.0`.
```

The example below uses the smallest model with 39M parameters. Consider using a [larger model](https://huggingface.co/models?other=whisper) to get better results.

```bash
ct2-transformers-converter --model openai/whisper-tiny --output_dir whisper-tiny-ct2
```

```python
import ctranslate2
import librosa
import transformers

# Load and resample the audio file.
audio, _ = librosa.load("audio.wav", sr=16000, mono=True)

# Compute the features of the first 30 seconds of audio.
processor = transformers.WhisperProcessor.from_pretrained("openai/whisper-tiny")
inputs = processor(audio, return_tensors="np", sampling_rate=16000)
features = ctranslate2.StorageView.from_array(inputs.input_features)

# Load the model on CPU.
model = ctranslate2.models.Whisper("whisper-tiny-ct2")

# Detect the language.
results = model.detect_language(features)
language, probability = results[0][0]
print("Detected language %s with probability %f" % (language, probability))

# Describe the task in the prompt.
# See the prompt format in https://github.com/openai/whisper.
prompt = processor.tokenizer.convert_tokens_to_ids(
    [
        "<|startoftranscript|>",
        language,
        "<|transcribe|>",
        "<|notimestamps|>",  # Remove this token to generate timestamps.
    ]
)

# Run generation for the 30-second window.
results = model.generate(features, [prompt])
transcription = processor.decode(results[0].sequences_ids[0])
print(transcription)
```

```{note}
This example only transcribes the first 30 seconds of audio. To transcribe longer files, you need to call `generate` on each 30-second window and aggregate the results. See the project [faster-whisper](https://github.com/guillaumekln/faster-whisper) for a complete transcription example using CTranslate2.
```
