# Code Completion

Code completion is a key feature offered by tabby in IDEs/extensions. By default, we use a relatively conservative configuration to accommodate low-end GPUs and smaller LLMs.
Tabby allows for more customized configuration by changing the `config.toml` file.

## Input/Output Limitations

Note that this configuration requires tuning of the model serving configuration as well (e.g., context length settings) and can vary significantly based on the model provider (e.g., llama.cpp, vLLM, TensorRT-LLM, etc).
Therefore, please only change these values if you have consulted with the model deployment vendor.

```toml
[completion]

# Maximum length of the input prompt, in UTF-8 characters, by default set to 1536.
max_input_length = 1536

# Maximum number of decoding tokens, by default set to 64.
max_decoding_tokens = 64
```
