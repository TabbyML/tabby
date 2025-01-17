# Fireworks

[Fireworks](https://app.fireworks.ai/) is a cloud platform that offers efficient model inference and deployment services,
providing cost-effective access to a variety of AI models through their API service,
including [Llama 2](https://fireworks.ai/models/fireworks/llama-v2-70b-chat),
[DeepSeek V3](https://fireworks.ai/models/fireworks/deepseek-v3),
[DeepSeek Coder](https://fireworks.ai/models/fireworks/deepseek-coder-v2-instruct) and other open-source models.

## Chat model

Fireworks provides an OpenAI-compatible chat API interface.

```toml title="~/.tabby/config.toml"
[model.chat.http]
kind = "openai/chat"
model_name = "accounts/fireworks/models/deepseek-v3"
api_endpoint = "https://api.fireworks.ai/inference/v1"
api_key = "your-api-key"
```

## Completion model

Fireworks does not offer completion models (FIM) through their API.

## Embeddings model

While Fireworks provides embedding model APIs, Tabby has not yet implemented a compatible client to interface with these APIs. Therefore, embedding functionality is currently not available through Tabby's integration with Fireworks.
