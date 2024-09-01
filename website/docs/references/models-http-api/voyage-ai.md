# Voyage AI

[Voyage AI](https://voyage.ai/) is a company that provides a range of embedding models. Tabby supports Voyage AI's models for embedding tasks.

Below is an example configuration:

```toml title="~/.tabby/config.toml"
[model.embedding.http]
kind = "voyage/embedding"
api_key = "..."
model_name = "voyage-code-2"
```