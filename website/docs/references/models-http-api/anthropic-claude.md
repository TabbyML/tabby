# Anthropic Claude

[Anthropic](https://www.anthropic.com/) is an AI research company that develops large language models, including the Claude family of models. While Tabby doesn't natively support Claude's API, you can access Claude models through an OpenAI-compatible API interface using [claude2openai](https://github.com/missuo/claude2openai) as a middleware.

## Chat model

After deploying the claude2openai middleware, you can access all Claude family models through an OpenAI-compatible chat API interface.

```toml title="~/.tabby/config.toml"
[model.chat.http]
kind = "openai/chat"
model_name = "claude-3-sonnet-20240229"
# Middleware endpoint (adjust host and port according to your deployment)
api_endpoint = "http://127.0.0.1:6600/v1"
api_key = "your-api-key"
```

## Completion model

Anthropic currently does not offer completion-specific API endpoints.

## Embeddings model

Anthropic currently does not provide embedding model APIs.
