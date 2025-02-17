# Tabby Model Specification

Tabby organizes the models within a directory.
This document provides an explanation of the necessary contents for supporting model serving.
A minimal Tabby model directory should include the following contents:

```
tabby.json
ggml/model-00001-of-00001.gguf
```

### tabby.json

This file provides meta information about the model. An example file appears as follows:

```json
{
    "prompt_template": "<PRE>{prefix}<SUF>{suffix}<MID>",
    "chat_template":  "<s>{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + '</s> ' }}{% endif %}{% endfor %}",
}
```

The **prompt_template** field is optional. When present, it is assumed that the model supports [FIM inference](https://arxiv.org/abs/2207.14255).

One example for the **prompt_template** is `<PRE>{prefix}<SUF>{suffix}<MID>`. In this format, `{prefix}` and `{suffix}` will be replaced with their corresponding values, and the entire prompt will be fed into the LLM.

The **chat_template** field is optional. When it is present, it is assumed that the model supports an instruct/chat-style interaction, and can be passed to `--chat-model`.

### ggml/

This directory contains binary files used by the [llama.cpp](https://github.com/ggml-org/llama.cpp) inference engine.
Tabby utilizes GGML for inference on `cpu`, `cuda` and `metal` devices.

Tabby saves GGUF model files in the format `model-{index}-of-{count}.gguf`, following the llama.cpp naming convention.
Please note that the index is 1-based,
by default, Tabby names a single file model as `model-00001-of-00001.gguf`.

For more details about GGUF models, please refer to the instructions in llama.cpp.
