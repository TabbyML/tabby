# Ollama

[ollama](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion) is a popular model provider that offers a local-first experience. It provides support for various models through HTTP APIs, including completion, chat, and embedding functionalities.

## Chat model

Ollama provides an OpenAI-compatible chat API interface.

```toml title="~/.tabby/config.toml"
[model.chat.http]
kind = "openai/chat"
model_name = "mistral:7b"
api_endpoint = "http://localhost:11434/v1"
```

## Completion model

Ollama offers a specialized completion API interface for code completion tasks.

```toml title="~/.tabby/config.toml"
[model.completion.http]
kind = "ollama/completion"
model_name = "codellama:7b"
api_endpoint = "http://localhost:11434"
prompt_template = "<PRE> {prefix} <SUF>{suffix} <MID>"  # Example prompt template for the CodeLlama model series.
```

## Embeddings model

Ollama provides embedding functionality through its HTTP API.

```toml title="~/.tabby/config.toml"
[model.embedding.http]
kind = "ollama/embedding"
model_name = "nomic-embed-text"
api_endpoint = "http://localhost:11434"
```
