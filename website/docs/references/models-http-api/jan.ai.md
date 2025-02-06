# Jan AI

[Jan](https://jan.ai/) is an open-source alternative to ChatGPT that runs entirely offline on your computer. It provides an OpenAI-compatible server interface that can be enabled through the Jan App's `Local API Server` UI.

## Chat model

Jan provides an OpenAI-compatible chat API interface.

```toml title="~/.tabby/config.toml"
[model.chat.http]
kind = "openai/chat"
model_name = "your_model"
api_endpoint = "http://localhost:1337/v1"
api_key = ""
```

## Completion model

Jan currently does not provide completion API support.

## Embeddings model

Jan currently does not provide embedding API support.
