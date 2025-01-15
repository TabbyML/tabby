# OpenAI

OpenAI is a leading AI company that has developed an extensive range of language models.
Tabby supports OpenAI's API specifications for chat, completion, and embedding tasks.

The OpenAI API is widely used and is also provided by other vendors,
such as vLLM, Nvidia NIM, and LocalAI.

Tabby continues to support the OpenAI Completion API specifications due to its widespread usage.

## Chat model

```toml title="~/.tabby/config.toml"
# Chat model
[model.chat.http]
kind = "openai/chat"
model_name = "gpt-4o"  # Please make sure to use a chat model, such as gpt-4o
api_endpoint = "https://api.openai.com/v1"   # DO NOT append the `/chat/completions` suffix
api_key = "secret-api-key"
```

## Completion model

OpenAI has designated its `/v1/completions` API for code completion as legacy,
and using OpenAI models for completion is no longer supported.

## Embeddings model

```toml title="~/.tabby/config.toml"
# Embedding model
[model.embedding.http]
kind = "openai/embedding"
model_name = "text-embedding-3-small"   # Please make sure to use a embedding model, such as text-embedding-3-small
api_endpoint = "https://api.openai.com/v1"  # DO NOT append the `/embeddings` suffix
api_key = "secret-api-key"
```
