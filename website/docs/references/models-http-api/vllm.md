# vLLM

[vLLM](https://docs.vllm.ai/en/stable/) is a fast and user-friendly library for LLM inference and serving.

vLLM offers an `OpenAI Compatible Server`, allowing us to use the OpenAI kinds for chat and embedding.
However, for completion, there are some differences in the implementation, so we should use the `vllm/completion` kind.

Below is an example

```toml title="~/.tabby/config.toml"
# Chat model
[model.chat.http]
kind = "openai/chat"
model_name = "your_model"
api_endpoint = "https://url_to_your_backend_or_service"
api_key = "secret-api-key"

# Embedding model
[model.embedding.http]
kind = "openai/embedding"
model_name = "your_model"
api_endpoint = "https://url_to_your_backend_or_service"
api_key = "secret-api-key"

# Completion model
[model.completion.http]
kind = "vllm/completion"
model_name = "your_model"
api_endpoint = "https://url_to_your_backend_or_service"
api_key = "secret-api-key"
```
