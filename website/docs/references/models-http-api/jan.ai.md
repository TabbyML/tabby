# Jan AI

[Jan](https://jan.ai/) is an open-source alternative to ChatGPT that runs entirely offline on your computer.

Jan can run a server that provides an OpenAI-equivalent chat API at https://localhost:1337,
allowing us to use the OpenAI kinds for chat.
To use the Jan Server, you need to enable it in the Jan App's `Local API Server` UI.

However, Jan does not yet provide API support for completion and embeddings.

Below is an example for chat:

```toml title="~/.tabby/config.toml"
# Chat model
[model.chat.http]
kind = "openai/chat"
model_name = "your_model"
api_endpoint = "http://localhost:1337/v1"
api_key = ""
```