# DeepInfra

[DeepInfra](https://deepinfra.com/) is a cloud platform providing efficient and scalable model inference services, offering access to various open-source models like [Llama 3](https://deepinfra.com/meta-llama/Llama-3.3-70B-Instruct), [Mixtral](https://deepinfra.com/mistralai/Mixtral-8x7B-Instruct-v0.1), and [Qwen](https://deepinfra.com/Qwen/Qwen2.5-Coder-32B-Instruct).

## Chat model

DeepInfra provides an OpenAI-compatible chat API interface.

```toml title="~/.tabby/config.toml"
[model.chat.http]
kind = "openai/chat"
model_name = "meta-llama/Llama-3.3-70B-Instruct"
api_endpoint = "https://api.deepinfra.com/v1/openai"
api_key = "your-api-key"
```

## Completion model

DeepInfra provides an OpenAI-compatible completion API interface.

```toml title="~/.tabby/config.toml"
[model.completion.http]
kind = "openai/completion"
model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
api_endpoint = "https://api.deepinfra.com/v1/openai"
api_key = "your-api-key"
```

## Embeddings model

DeepInfra also provides an OpenAI-compatible embeddings API interface.

```toml title="~/.tabby/config.toml"
[model.embedding.http]
kind = "openai/embedding"
model_name = "BAAI/bge-base-en-v1.5"
api_endpoint = "https://api.deepinfra.com/v1/openai"
api_key = "your-api-key"
```
