# Code Completion

Code completion is a key feature offered by Tabby in IDEs/extensions. Tabby allows for more customized configuration by modifying the `config.toml` file in the `[completion]` section.

## Input / Output 

This configuration requires tuning of the model serving configuration as well (e.g., context length settings) and can vary significantly based on the model provider (e.g., llama.cpp, vLLM, TensorRT-LLM, etc).
Therefore, please only modify these values after consulting with the model deployment vendor.

```toml
[completion]

# Maximum length of the input prompt, in UTF-8 characters. The default value is set to 1536.
max_input_length = 1536

# Maximum number of decoding tokens. The default value is set to 64.
max_decoding_tokens = 64
```

The default value is set conservatively to accommodate local GPUs and smaller LLMs.

