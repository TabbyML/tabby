# Ollama

[ollama](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion) is a popular model provider that offers a local-first experience, powered by llama.cpp.

Tabby supports the ollama HTTP API for completion, chat, and embedding models.

```toml title="~/.tabby/config.toml"
# Completion model
[model.completion.http]
kind = "ollama/completion"
model_name = "codellama:7b"
api_endpoint = "http://localhost:11434"
prompt_template = "<PRE> {prefix} <SUF>{suffix} <MID>"  # Example prompt template for the CodeLlama model series.

# Chat model
[model.chat.http]
kind = "openai/chat"
model_name = "mistral:7b"
api_endpoint = "http://localhost:11434/v1"

# Embedding model
[model.embedding.http]
kind = "ollama/embedding"
model_name = "nomic-embed-text"
api_endpoint = "http://localhost:11434"
```
