# MiniMax

[MiniMax](https://www.minimax.io/) is an AI company that develops large language models for general tasks. Their models include [MiniMax-M2.7](https://platform.minimax.io/docs/api-reference/text-openai-api) for high-performance language understanding and generation.

## Chat model

MiniMax provides an OpenAI-compatible chat API interface. Tabby includes a dedicated `minimax/chat` kind that handles MiniMax-specific constraints such as temperature clamping.

```toml title="~/.tabby/config.toml"
[model.chat.http]
kind = "minimax/chat"
model_name = "MiniMax-M2.7"
api_endpoint = "https://api.minimax.io/v1"
api_key = "your-minimax-api-key"
```

You can also configure multi-model support to switch between available models:

```toml title="~/.tabby/config.toml"
[model.chat.http]
kind = "minimax/chat"
model_name = "MiniMax-M2.7"
supported_models = ["MiniMax-M2.7", "MiniMax-M2.7-highspeed"]
api_endpoint = "https://api.minimax.io/v1"
api_key = "your-minimax-api-key"
```

For users in mainland China, use the domestic endpoint:

```toml title="~/.tabby/config.toml"
[model.chat.http]
kind = "minimax/chat"
model_name = "MiniMax-M2.7"
api_endpoint = "https://api.minimaxi.com/v1"
api_key = "your-minimax-api-key"
```

## Completion model

MiniMax does not currently offer a dedicated completion (FIM) API endpoint.

## Embeddings model

MiniMax does not currently offer embedding model APIs.

## Available Models

| Model | Context Length | Max Output |
|---|---|---|
| `MiniMax-M2.7` | 204K | 192K |
| `MiniMax-M2.7-highspeed` | 204K | 192K |

For the latest model list and pricing, visit [MiniMax Platform](https://platform.minimax.io/).
