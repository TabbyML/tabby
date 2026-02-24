# OpenRouter

[OpenRouter](https://openrouter.ai/) provides unified access to multiple AI models through an OpenAI API compatible RESTful endpoint, including models from OpenAI, Anthropic, Google, and Meta.

## Chat Model

OpenRouter provides an OpenAI-compatible chat API interface.

```toml title="~/.tabby/config.toml"
[[endpoints]]
name = "openrouter_chat"
api_route = "https://openrouter.ai/api"
headers = {
  Authorization = "Bearer your-api-key"
}
metadata = {
  pochi = {
    use_case = "chat",
    provider = "openai",
    models = [
      { name = "openai/gpt-4", context_window = 128000 }
    ]
  }
}
```

## Completion Model

OpenRouter does not offer completion models (FIM) through their API.

## Supported Models

For a complete list of supported models, visit [OpenRouter's Model List](https://openrouter.ai/models).
