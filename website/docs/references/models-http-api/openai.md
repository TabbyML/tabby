# OpenAI

OpenAI is a leading AI company that has developed a range of language models. Tabby supports OpenAI's models for chat and embedding tasks.

Tabby also supports its legacy `/v1/completions` API for code completion, although **OpenAI itself no longer supports it**; it is still the API offered by some other vendors, such as (vLLM, Nvidia NIM, LocalAI, ...).

Below is an example configuration:

```toml title="~/.tabby/config.toml"
# Completion model
[model.completion.http]
kind = "openai/completion"
model_name = "your_model"
api_endpoint = "https://url_to_your_backend_or_service"
api_key = "secret-api-key"

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
```
