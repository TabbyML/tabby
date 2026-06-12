# EmpirioLabs

[EmpirioLabs](https://empiriolabs.ai/) is a multi-model API platform offering open and proprietary models (Qwen, DeepSeek, GLM, Kimi, MiniMax, Gemma, and more) through an OpenAI-compatible endpoint with pay-as-you-go pricing. The full model catalog with per-model context windows and pricing is at [empiriolabs.ai/models](https://empiriolabs.ai/models).

## Chat Model

EmpirioLabs provides an OpenAI-compatible chat API interface.

```toml title="~/.tabby/config.toml"
[model.chat.http]
kind = "openai/chat"
model_name = "qwen3-7-plus"
api_endpoint = "https://api.empiriolabs.ai/v1"
api_key = "your-api-key"
```

You can also configure multi-model support to switch between available models:

```toml title="~/.tabby/config.toml"
[model.chat.http]
kind = "openai/chat"
model_name = "qwen3-7-plus"
supported_models = [
  "qwen3-7-plus",
  "deepseek-v4-pro",
  "glm-5-1",
  "kimi-k2-7-code"
]
api_endpoint = "https://api.empiriolabs.ai/v1"
api_key = "your-api-key"
```

## Completion Model

EmpirioLabs does not currently offer a dedicated completion (FIM) API endpoint.

## Embeddings Model

EmpirioLabs provides an OpenAI-compatible embeddings API interface.

```toml title="~/.tabby/config.toml"
[model.embedding.http]
kind = "openai/embedding"
model_name = "text-embedding-v4"
api_endpoint = "https://api.empiriolabs.ai/v1"
api_key = "your-api-key"
```
