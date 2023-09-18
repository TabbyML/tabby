# Tabby Model Specification (Unstable)

Tabby organizes the model within a directory. This document provides an explanation of the necessary contents for supporting model serving. An example model directory can be found at https://huggingface.co/TabbyML/StarCoder-1B

The minimal Tabby model directory should include the following contents:

```
ctranslate2/
ggml/
tabby.json
tokenizer.json
```

### tabby.json

This file provides meta information about the model. An example file appears as follows:

```js
{
    "auto_model": "AutoModelForCausalLM",
    "prompt_template": "<PRE>{prefix}<SUF>{suffix}<MID>"
}
```

The **auto_model** field can have one of the following values:
- `AutoModelForCausalLM`: This represents a decoder-only style language model, such as GPT or Llama.
- `AutoModelForSeq2SeqLM`: This represents an encoder-decoder style language model, like T5.

The **prompt_template** field is optional. When present, it is assumed that the model supports [FIM inference](https://arxiv.org/abs/2207.14255).

One example for the **prompt_template** is `<PRE>{prefix}<SUF>{suffix}<MID>`. In this format, `{prefix}` and `{suffix}` will be replaced with their corresponding values, and the entire prompt will be fed into the LLM.

### tokenizer.json
This is the standard fast tokenizer file created using Hugging Face Tokenizers. Most Hugging Face models already come with it in repository.

### ctranslate2/
This directory contains binary files used by the [ctranslate2](https://github.com/OpenNMT/CTranslate2) inference engine. Tabby utilizes ctranslate2 for inference on both `cpu` and `cuda` devices.

With the [python package](https://pypi.org/project/ctranslate2) installed, you can acquire this directory by executing the following command in the HF model directory:

```bash
ct2-transformers-converter --model ./ --output_dir ctranslate2 --quantization=float16
```

*Note that the model itself must be compatible with ctranslate2.*

### ggml/
This directory contains binary files used by the [llama.cpp](https://github.com/ggerganov/llama.cpp) inference engine. Tabby utilizes ctranslate2 for inference on the `metal` device.

Currently, only `q8_0.gguf` in this directory is in use. You can refer to the instructions in llama.cpp to learn how to acquire it.
