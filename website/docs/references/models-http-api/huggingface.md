# Hugging Face

[Hugging Face Inference Providers](https://huggingface.co/inference-providers) offers access to open-source models from multiple providers through a unified API. Inference Providers automatically routes requests to the best available provider, supporting models like [Qwen 2.5 Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct), [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-R1), and [Llama](https://huggingface.co/meta-llama).

## Chat model

Hugging Face Inference Providers provides an OpenAI-compatible chat API interface.

```toml title="~/.tabby/config.toml"
[model.chat.http]
kind = "openai/chat"
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
api_endpoint = "https://router.huggingface.co/v1"
api_key = "your-hf-token"
```

### Recommended models

The following models from Tabby's recommended list are available through Inference Providers:

- `Qwen/Qwen2.5-Coder-1.5B-Instruct` - Lightweight model for basic tasks
- `Qwen/Qwen2.5-Coder-7B-Instruct` - Balanced performance and efficiency
- `Qwen/Qwen2.5-Coder-14B-Instruct` - Higher capability model
- `Qwen/Qwen2.5-Coder-32B-Instruct` - Most powerful model with multiple provider options

For a complete list of available models, visit the [Inference Providers documentation](https://huggingface.co/docs/inference-providers).

## Completion model

Hugging Face Inference Providers does not offer completion models (FIM) through their OpenAI-compatible API. For code completion, use a local model with Tabby.

## Embeddings model

Embedding models are not currently available through Hugging Face Inference Providers. For embeddings, use a local model with Tabby.