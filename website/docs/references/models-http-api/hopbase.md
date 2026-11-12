# HopBase

[HopBase](https://hop-base.com/) is an API gateway that serves models such as GPT, Claude and Gemini alongside GLM, DeepSeek, Qwen and Kimi through one OpenAI API compatible base URL.

## Chat model

HopBase provides an OpenAI-compatible chat API interface. Use model names as listed by `GET https://api.hop-base.com/v1/models`, which returns the models your API key can call. With `supported_models`, you can switch between them in the chat interface.

```toml title="~/.tabby/config.toml"
[model.chat.http]
kind = "openai/chat"
model_name = "gpt-6.1-sol"
supported_models = ["gpt-6.1-sol", "glm-5.3", "claude-sonnet-5-5"] # Optional, for multi-model support
api_endpoint = "https://api.hop-base.com/v1"   # DO NOT append the `/chat/completions` suffix
api_key = "your-api-key"
```

## Completion model

HopBase does not offer completion models (FIM) through its API. Configure the completion model with a local model or another provider, for example:

```toml title="~/.tabby/config.toml"
[model.completion.local]
model_id = "StarCoder2-3B"
```

## Embeddings model

HopBase does not offer embeddings models through its API. Keep the default local embedding model, or configure another provider:

```toml title="~/.tabby/config.toml"
[model.embedding.local]
model_id = "Nomic-Embed-Text"
```

## Supported Models

For a list of available models, visit [HopBase models](https://hop-base.com/models). API keys can be created in the [HopBase console](https://api.hop-base.com/login).
