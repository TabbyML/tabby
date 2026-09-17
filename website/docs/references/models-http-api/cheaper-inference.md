# Cheaper Inference

[Cheaper Inference](https://cheaperinference.com/) is an OpenAI-compatible gateway that serves chat models from several makers (Anthropic, OpenAI, Google, xAI, DeepSeek, Moonshot, Z.ai, Alibaba) behind one endpoint and one API key, at or below the maker's list price. Model ids are the maker's own, with no vendor prefix.

## Chat model

Cheaper Inference provides an OpenAI-compatible chat API interface.

```toml title="~/.tabby/config.toml"
[model.chat.http]
kind = "openai/chat"
model_name = "claude-sonnet-5"  # Can be any chat model the gateway serves
api_endpoint = "https://api.cheaperinference.com/v1"
api_key = "your-api-key"
```

## Completion model

Cheaper Inference serves chat models only; its catalogue has no fill-in-the-middle model, so point `model.completion` at a provider that offers one.

## Embeddings model

Cheaper Inference does not offer embeddings models - `POST /v1/embeddings` returns 404.

## Supported models

`GET /v1/models` on the gateway lists every model it serves, with per-model prices, context and output limits.
