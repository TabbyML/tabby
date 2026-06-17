# Pinstripes

[Pinstripes](https://pinstripes.io) is a cloud inference platform providing fast, cost-effective access to popular open-source models through an OpenAI-compatible API.

## Chat model

Pinstripes provides an OpenAI-compatible chat API interface.

```toml title="~/.tabby/config.toml"
[model.chat.http]
kind = "openai/chat"
model_name = "ps/deepseek-v4-flash"
api_endpoint = "https://pinstripes.io/v1"
api_key = "your-api-key"
```

Available chat models include:

- `ps/deepseek-v4-flash` — $0.10/M tokens
- `ps/glm-4.5-air` — $0.125/M tokens
- `ps/qwen3-35b` — $0.14/M tokens
- `ps/minimax-m2.7` — $0.255/M tokens

Set `PINSTRIPES_API_KEY` in your environment, or pass the key directly via `api_key`.

## Completion model

Pinstripes does not currently offer FIM completion models through their API.

## Embeddings model

Pinstripes does not currently offer embedding models through their API.
