# llamafile

[llamafile](https://github.com/Mozilla-Ocho/llamafile) is a Mozilla Builders project that allows you to distribute and run LLMs with a single file. It embeds a llama.cpp server and provides an OpenAI API-compatible chat-completions endpoint, allowing us to use the `openai/chat`, `llama.cpp/completion`, and `llama.cpp/embedding` types.

By default, llamafile uses port `8080`, which conflicts with Tabby's default port. It is recommended to run llamafile with the `--port` option to serve on a different port, such as `8081`. For embeddings functionality, you need to run llamafile with both the `--embedding` and `--port` options.

## Chat model

llamafile provides an OpenAI-compatible chat API interface. Note that the endpoint URL must include the `v1` suffix.

```toml title="~/.tabby/config.toml"
[model.chat.http]
kind = "openai/chat"  # llamafile uses openai/chat kind
model_name = "your_model"
api_endpoint = "http://localhost:8081/v1"  # Please add and conclude with the `v1` suffix
api_key = ""
```

## Completion model

llamafile uses llama.cpp's completion API interface. Note that the endpoint URL should NOT include the `v1` suffix.

```toml title="~/.tabby/config.toml"
[model.completion.http]
kind = "llama.cpp/completion"
model_name = "your_model"
api_endpoint = "http://localhost:8081"  # DO NOT append the `v1` suffix
api_key = "secret-api-key"
prompt_template = "<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>" # Example prompt template for the Qwen2.5 Coder model series.
```

## Embeddings model

llamafile provides embedding functionality via llama.cpp's API interface,
but it utilizes the API interface defined prior to version b4356.
Therefore, we should use the kind `llama.cpp/before_b4356_embedding`.

Note that the endpoint URL should NOT include the `v1` suffix.

```toml title="~/.tabby/config.toml"
[model.embedding.http]
kind = "llama.cpp/before_b4356_embedding"
model_name = "your_model"
api_endpoint = "http://localhost:8082"  # DO NOT append the `v1` suffix
api_key = ""
```
