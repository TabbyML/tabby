# DeepSeek

[DeepSeek](https://www.deepseek.com/) is an AI company that develops large language models specialized in coding and general tasks. Their models include [DeepSeek V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) for general tasks and [DeepSeek Coder](https://huggingface.co/collections/deepseek-ai/deepseekcoder-v2-666bf4b274a5f556827ceeca) specifically optimized for programming tasks.

## Chat model

DeepSeek provides an OpenAI-compatible chat API interface.

```toml title="~/.tabby/config.toml"
[[endpoints]]
name = "deepseek_chat"
api_route = "https://api.deepseek.com"
headers = {
  Authorization = "Bearer your-api-key"
}
metadata = {
  pochi = {
    use_case = "chat",
    provider = "openai",
    models = [
      { name = "deepseek-chat", context_window = 128000 }
    ]
  }
}
```

<!-- FIXME(wei) update Completion config-->
## Completion model

DeepSeek offers a specialized completion API interface for code completion tasks.

```toml title="~/.tabby/config.toml"
[[endpoints]]
name = "deepseek_completion"
api_route = "https://api.deepseek.com/beta"
headers = {
  Authorization = "Bearer your-api-key"
}
metadata = {
  pochi = {
    use_case = "completion",
    provider = "openai",
    models = [
      { name = "deepseek-coder", context_window = 128000 }
    ]
  }
}
```
