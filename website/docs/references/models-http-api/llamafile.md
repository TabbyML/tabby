# llamafile

[llamafile](https://github.com/Mozilla-Ocho/llamafile)
is a Mozilla Builders project that allows you to distribute and run LLMs with a single file.

llamafile embeds a llama.cpp server and provides an OpenAI API-compatible chat-completions endpoint,
allowing us to use the `openai/chat`, `llama.cpp/completion`, and `llama.cpp/embedding` types.

By default, llamafile uses port `8080`, which is also used by Tabby.
Therefore, it is recommended to run llamafile with the `--port` option to serve on a different port, such as `8081`.

For embeddings, the embedding endpoint is no longer supported in the standard llamafile server,
so you need to run llamafile with the `--embedding` and `--port` options.

Below is an example configuration:

```toml title="~/.tabby/config.toml"
# Chat model
[model.chat.http]
kind = "openai/chat"  # llamafile uses openai/chat kind
model_name = "your_model"
api_endpoint = "http://localhost:8081/v1"  # Please add and conclude with the `v1` suffix
api_key = ""

# Completion model
[model.completion.http]
kind = "llama.cpp/completion"   # llamafile uses llama.cpp/completion kind
model_name = "your_model"
api_endpoint = "http://localhost:8081"  # Do NOT append the `v1` suffix
api_key = "secret-api-key"
prompt_template = "<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>" # Example prompt template for the Qwen2.5 Coder model series.

# Embedding model
[model.embedding.http]
kind = "llama.cpp/embedding"  # llamafile uses llama.cpp/embedding kind
model_name = "your_model"
api_endpoint = "http://localhost:8082"  # Do NOT append the `v1` suffix
api_key = ""
```
