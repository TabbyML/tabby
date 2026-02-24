# Fireworks

[Fireworks](https://app.fireworks.ai/) is a cloud platform that offers efficient model inference and deployment services,
providing cost-effective access to a variety of AI models through their API service,
including [Llama 2](https://fireworks.ai/models/fireworks/llama-v2-70b-chat),
[DeepSeek V3](https://fireworks.ai/models/fireworks/deepseek-v3),
[DeepSeek Coder](https://fireworks.ai/models/fireworks/deepseek-coder-v2-instruct) and other open-source models.

## Chat model

Fireworks provides an OpenAI-compatible chat API interface.

```toml title="~/.tabby/config.toml"
[[endpoints]]
name = "fireworks_chat"
api_route = "https://api.fireworks.ai/inference"
headers = {
  Authorization = "Bearer your-api-key"
}
metadata = {
  pochi = {
    use_case = "chat",
    provider = "openai",
    models = [
      { name = "accounts/fireworks/models/deepseek-v3", context_window = 128000 }
    ]
  }
}
```

## Completion model

Fireworks does not offer completion models (FIM) through their API.
