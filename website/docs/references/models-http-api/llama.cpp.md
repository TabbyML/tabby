# llama.cpp

[llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md#api-endpoints) is a popular C++ library for serving gguf-based models. It provides a server implementation that supports completion, chat, and embedding functionalities through HTTP APIs.

## Chat model

llama.cpp provides an OpenAI-compatible chat API interface.

```toml title="~/.tabby/config.toml"
[model.chat.http]
kind = "openai/chat"
api_endpoint = "http://localhost:8888"
```

## Completion model

llama.cpp offers a specialized completion API interface for code completion tasks.

```toml title="~/.tabby/config.toml"
[model.completion.http]
kind = "llama.cpp/completion"
api_endpoint = "http://localhost:8888"
prompt_template = "<PRE> {prefix} <SUF>{suffix} <MID>"  # Example prompt template for the CodeLlama model series.
```

## Embeddings model

llama.cpp provides embedding functionality through its HTTP API.

```toml title="~/.tabby/config.toml"
[model.embedding.http]
kind = "llama.cpp/embedding"
api_endpoint = "http://localhost:8888"
```
