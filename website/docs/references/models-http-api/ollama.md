# Ollama

[ollama](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion) is a popular model provider that offers a local-first experience. It provides support for various models through HTTP APIs.

## Chat model

Ollama provides an OpenAI-compatible chat API interface.

```toml title="~/.tabby/config.toml"
[[endpoints]]
name = "ollama_chat"
api_route = "http://localhost:11434"
metadata = {
  pochi = {
    use_case = "chat",
    provider = "openai",
    models = [
      { name = "mistral:7b", context_window = 4096 }
    ]
  }
}
```

<!-- FIXME(wei) update Completion config-->
## Completion model

Ollama offers a specialized completion API interface for code completion tasks.

```toml title="~/.tabby/config.toml"
[[endpoints]]
name = "ollama_completion"
api_route = "http://localhost:11434/v1"
metadata = {
  pochi = {
    use_case = "completion",
    provider = "openai",
    models = [
      { name = "codellama:7b", context_window = 4096 }
    ]
  }
}
```
