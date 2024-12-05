# OpenAI

OpenAI is a leading AI company that has developed an extensive range of language models.
Tabby supports OpenAI's API specifications for chat, completion, and embedding tasks.

The OpenAI API is widely used and is also provided by other vendors,
such as vLLM, Nvidia NIM, and LocalAI.

OpenAI has designated its `/v1/completions` API for code completion as legacy,
and **OpenAI itself no longer supports it**.

Tabby continues to support the OpenAI Completion API specifications due to its widespread usage.

```toml title="~/.tabby/config.toml"
# Chat model
[model.chat.http]
kind = "openai/chat"
model_name = "gpt-3.5-turbo"
api_endpoint = "https://api.openai.com/v1"
api_key = "secret-api-key"

# Embedding model
[model.embedding.http]
kind = "openai/embedding"
model_name = "text-embedding-3-small"
api_endpoint = "https://api.openai.com/v1"
api_key = "secret-api-key"

# Completion model
[model.completion.http]
kind = "openai/completion"
model_name = "your_model"
api_endpoint = "https://url_to_your_backend_or_service"
api_key = "secret-api-key"
```
