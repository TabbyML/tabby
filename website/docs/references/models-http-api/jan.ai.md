# Jan AI

[Jan](https://jan.ai/) is an open-source alternative to ChatGPT that runs entirely offline on your computer. It provides an OpenAI-compatible server interface that can be enabled through the Jan App's `Local API Server` UI.

## Chat model

Jan provides an OpenAI-compatible chat API interface.

```toml title="~/.tabby/config.toml"
[[endpoints]]
name = "jan_chat"
api_route = "http://localhost:1337"
metadata = {
  pochi = {
    use_case = "chat",
    provider = "openai",
    models = [
      { name = "your_model", context_window = 4096 }
    ]
  }
}
```

## Completion model

Jan currently does not provide completion API support.
