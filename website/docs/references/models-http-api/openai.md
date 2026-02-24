# OpenAI

OpenAI is a leading AI company that has developed an extensive range of language models. Their API specifications have become a de facto standard, also implemented by other vendors such as vLLM, Nvidia NIM, and LocalAI.

## Chat model

OpenAI provides a comprehensive chat API interface.

```toml title="~/.tabby/config.toml"
[[endpoints]]
name = "openai"
api_route = "https://api.openai.com"
headers = {
  Authorization = "Bearer your-api-key"
}
metadata = {
  pochi = {
    use_case = "chat",
    provider = "openai",
    models = [
      { name = "gpt-4o", context_window = 128000 }
    ]
  }
}
```

## Completion model

OpenAI doesn't offer models for completions (FIM), its `/v1/completions` API has been deprecated.
