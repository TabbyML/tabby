# Mistral AI

[Mistral](https://mistral.ai/) is a platform that provides a suite of AI models specialized in various tasks, including code generation and natural language processing. Their models are known for high performance and efficiency in both code completion and chat interactions.

## Chat model

Mistral provides a specialized chat API interface.

```toml title="~/.tabby/config.toml"
[[endpoints]]
name = "mistral_chat"
api_route = "https://api.mistral.ai"
headers = {
  Authorization = "Bearer your-api-key"
}
metadata = {
  pochi = {
    use_case = "chat",
    provider = "openai",
    models = [
      { name = "codestral-latest", context_window = 32000 }
    ]
  }
}
```

<!-- FIXME(wei) update Completion config-->
## Completion model

Mistral offers a dedicated completion API interface for code completion tasks.

```toml title="~/.tabby/config.toml"
[[endpoints]]
name = "mistral_completion"
api_route = "https://api.mistral.ai/v1"
headers = {
  Authorization = "Bearer your-api-key"
}
metadata = {
  pochi = {
    use_case = "completion",
    provider = "mistral",
    models = [
      { name = "codestral-latest", context_window = 32000 }
    ]
  }
}
```
