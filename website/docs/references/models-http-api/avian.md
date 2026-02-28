# Avian

[Avian](https://avian.io/) is an inference API provider offering access to frontier open-source models through an OpenAI-compatible endpoint. Available models include DeepSeek V3.2, Kimi K2.5, GLM-5, and MiniMax M2.5.

## Chat Model

Avian provides an OpenAI-compatible chat API interface.

```toml title="~/.tabby/config.toml"
[model.chat.http]
kind = "openai/chat"
model_name = "deepseek/deepseek-v3.2"
api_endpoint = "https://api.avian.io/v1"
api_key = "your-api-key"
```

You can also configure multi-model support to switch between available models:

```toml title="~/.tabby/config.toml"
[model.chat.http]
kind = "openai/chat"
model_name = "deepseek/deepseek-v3.2"
supported_models = [
  "deepseek/deepseek-v3.2",
  "moonshotai/kimi-k2.5",
  "z-ai/glm-5",
  "minimax/minimax-m2.5"
]
api_endpoint = "https://api.avian.io/v1"
api_key = "your-api-key"
```

## Completion Model

Avian does not currently offer a dedicated completion (FIM) API endpoint.

## Embeddings Model

Avian does not currently offer embedding model APIs.

## Available Models

| Model | Context Length | Max Output |
|---|---|---|
| `deepseek/deepseek-v3.2` | 164K | 65K |
| `moonshotai/kimi-k2.5` | 131K | 8K |
| `z-ai/glm-5` | 131K | 16K |
| `minimax/minimax-m2.5` | 1M | 1M |

For the latest model list and pricing, visit [Avian](https://avian.io/).
