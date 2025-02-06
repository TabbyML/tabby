# Mistral AI

[Mistral](https://mistral.ai/) is a platform that provides a suite of AI models specialized in various tasks, including code generation and natural language processing. Their models are known for high performance and efficiency in both code completion and chat interactions.

## Chat model

Mistral provides a specialized chat API interface.

```toml title="~/.tabby/config.toml"
[model.chat.http]
kind = "mistral/chat"
model_name = "codestral-latest"
api_endpoint = "https://api.mistral.ai/v1"
api_key = "your-api-key"
```

## Completion model

Mistral offers a dedicated completion API interface for code completion tasks.

```toml title="~/.tabby/config.toml"
[model.completion.http]
kind = "mistral/completion"
model_name = "codestral-latest"
api_endpoint = "https://api.mistral.ai"
api_key = "your-api-key"
```

## Embeddings model

Mistral currently does not provide embedding model APIs.
