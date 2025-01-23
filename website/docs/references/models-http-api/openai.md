# OpenAI

OpenAI is a leading AI company that has developed an extensive range of language models. Their API specifications have become a de facto standard, also implemented by other vendors such as vLLM, Nvidia NIM, and LocalAI.

## Chat model

OpenAI provides a comprehensive chat API interface. Note: Do not append the `/chat/completions` suffix to the API endpoint.

```toml title="~/.tabby/config.toml"
[model.chat.http]
kind = "openai/chat"
model_name = "gpt-4o"  # Please make sure to use a chat model, such as gpt-4o
api_endpoint = "https://api.openai.com/v1"   # DO NOT append the `/chat/completions` suffix
api_key = "your-api-key"
```

## Completion model

OpenAI doesn't offer models for completions (FIM), its `/v1/completions` API has been deprecated.

## Embeddings model

OpenAI provides powerful embedding models through their API interface. Note: Do not append the `/embeddings` suffix to the API endpoint.

```toml title="~/.tabby/config.toml"
[model.embedding.http]
kind = "openai/embedding"
model_name = "text-embedding-3-small"  # Please make sure to use a embedding model, such as text-embedding-3-small
api_endpoint = "https://api.openai.com/v1"  # DO NOT append the `/embeddings` suffix
api_key = "your-api-key"
```
