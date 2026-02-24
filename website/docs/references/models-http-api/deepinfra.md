# DeepInfra

[DeepInfra](https://deepinfra.com/) is a cloud platform providing efficient and scalable model inference services, offering access to various open-source models like [Llama 3](https://deepinfra.com/meta-llama/Llama-3.3-70B-Instruct), [Mixtral](https://deepinfra.com/mistralai/Mixtral-8x7B-Instruct-v0.1), and [Qwen](https://deepinfra.com/Qwen/Qwen2.5-Coder-32B-Instruct).

## Chat model

DeepInfra provides an OpenAI-compatible chat API interface.

```toml title="~/.tabby/config.toml"
[[endpoints]]
name = "deepinfra_chat"
api_route = "https://api.deepinfra.com/v1/openai"
headers = {
  Authorization = "Bearer your-api-key"
}
metadata = {
  pochi = {
    use_case = "chat",
    provider = "openai",
    models = [
      { name = "meta-llama/Llama-3.3-70B-Instruct", context_window = 128000 }
    ]
  }
}
```

<!-- FIXME(wei) update Completion config-->
## Completion model

DeepInfra provides an OpenAI-compatible completion API interface.

```toml title="~/.tabby/config.toml"
[[endpoints]]
name = "deepinfra_completion"
api_route = "https://api.deepinfra.com/v1/openai"
headers = {
  Authorization = "Bearer your-api-key"
}
metadata = {
  pochi = {
    use_case = "completion",
    provider = "openai",
    models = [
      { name = "Qwen/Qwen2.5-Coder-32B-Instruct", context_window = 32000 }
    ]
  }
}
```
