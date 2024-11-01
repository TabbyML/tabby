# llama.cpp

[llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md#api-endpoints) is a popular C++ library for serving gguf-based models.

Tabby supports the llama.cpp HTTP API for completion, chat, and embedding models.

```toml title="~/.tabby/config.toml"
# Completion model
[model.completion.http]
kind = "llama.cpp/completion"
api_endpoint = "http://localhost:8888"
prompt_template = "<PRE> {prefix} <SUF>{suffix} <MID>"  # Example prompt template for the CodeLlama model series.

# Chat model
[model.chat.http]
kind = "openai/chat"
api_endpoint = "http://localhost:8888"

# Embedding model
[model.embedding.http]
kind = "llama.cpp/embedding"
api_endpoint = "http://localhost:8888"
```
