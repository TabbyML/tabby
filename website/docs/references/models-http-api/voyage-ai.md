# Voyage AI

[Voyage AI](https://voyage.ai/) is a company that provides a range of embedding models. Tabby supports Voyage AI's models for embedding tasks.

## Embeddings model

Voyage AI provides specialized embedding models through their API interface.

```toml title="~/.tabby/config.toml"
[model.embedding.http]
kind = "voyage/embedding"
model_name = "voyage-code-2"
api_key = "your-api-key"
```
