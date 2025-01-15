# OpenRouter

[OpenRouter](https://openrouter.ai/) provides unified access to multiple AI models through an OpenAI API compatible RESTful endpoint, including models from OpenAI, Anthropic, Google, and Meta.

## Chat Model

OpenRouter provides an OpenAI-compatible chat API interface.

```toml title="~/.tabby/config.toml"
[model.chat.http]
kind = "openai/chat"
model_name = "openai/gpt-4"  # Can be any model from https://openrouter.ai/models
api_endpoint = "https://openrouter.ai/api/v1"
api_key = "your-api-key"
```

## Completion Model

OpenRouter does not offer completion models (FIM) through their API.

## Embeddings Model

OpenRouter does not offer embeddings models through their API.

## Supported Models

For a complete list of supported models, visit [OpenRouter's Model List](https://openrouter.ai/models).
