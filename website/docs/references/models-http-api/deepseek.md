# DeepSeek

[DeepSeek](https://www.deepseek.com/) is an AI company that develops large language models specialized in coding and general tasks. Their models include [DeepSeek V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) for general tasks and [DeepSeek Coder](https://huggingface.co/collections/deepseek-ai/deepseekcoder-v2-666bf4b274a5f556827ceeca) specifically optimized for programming tasks.

## Chat model

DeepSeek provides an OpenAI-compatible chat API interface.

```toml title="~/.tabby/config.toml"
[model.chat.http]
kind = "openai/chat"
model_name = "your_model"
api_endpoint = "https://api.deepseek.com/v1"
api_key = "your-api-key"
```

## Completion model

DeepSeek offers a specialized completion API interface for code completion tasks.

```toml title="~/.tabby/config.toml"
[model.completion.http]
kind = "deepseek/completion"
model_name = "your_model"
api_endpoint = "https://api.deepseek.com/beta"
api_key = "your-api-key"
```

## Embeddings model

DeepSeek currently does not provide embedding model APIs.
