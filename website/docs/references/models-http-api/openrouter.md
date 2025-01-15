# OpenRouter

[OpenRouter](https://openrouter.ai/) provides unified access to multiple AI models through an OpenAI API compatible RESTful endpoint, including models from OpenAI, Anthropic, Google, and Meta.

## Chat Model

OpenRouter provides an OpenAI-compatible chat API interface.

```toml title="~/.tabby/config.toml"
[model.chat.http]
kind = "openai/chat"
model_name = "openai/gpt-4"  # Can be any model from https://openrouter.ai/models
api_endpoint = "https://openrouter.ai/api/v1"
api_key = "your-api-key"
```

## Completion Model

OpenRouter does not offer completion models (FIM) through their API.

## Embeddings Model

OpenRouter does not offer embeddings models through their API.

## Supported Models

OpenRouter supports a wide range of models. Here are some commonly used ones:

| Model Name                  | Provider  | Description           |
| --------------------------- | --------- | --------------------- |
| openai/gpt-4                | OpenAI    | GPT-4 8K context      |
| openai/gpt-3.5-turbo        | OpenAI    | GPT-3.5 Turbo 16K     |
| anthropic/claude-2          | Anthropic | Claude 2 100K context |
| google/palm-2-chat-bison    | Google    | PaLM 2 Chat           |
| meta-llama/llama-2-70b-chat | Meta      | LLaMA 2 70B Chat      |

For a complete list of supported models, visit [OpenRouter's Model List](https://openrouter.ai/models).
