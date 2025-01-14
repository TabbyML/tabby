# Azure OpenAI

[Azure OpenAI](https://azure.microsoft.com/products/ai-services/openai-service) is a cloud-based service that provides Azure customers with access to OpenAI's powerful language models including GPT-4, GPT-3.5, and various embedding models.

## Chat model

It supports various GPT series chat models through an Azure OpenAI-compatible API interface.

```toml title="~/.tabby/config.toml"
[model.chat.http]
kind = "azure/chat"
model_name = "gpt-4o-mini"
api_endpoint = "https://xxx.openai.azure.com"
api_key = "your-api-key"
```

## Completion model

Azure OpenAI currently does not offer completion-specific API endpoints.

## Embeddings model

It supports text-embedding-3-small, text-embedding-3-large and other embedding models through an Azure OpenAI-compatible API interface.

```toml title="~/.tabby/config.toml"
[model.embedding.http]
kind = "azure/embedding"
model_name = "text-embedding-3-small"
api_endpoint = "https://xxx.openai.azure.com"
api_key = "your-api-key"
```
