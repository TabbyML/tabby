# Mistral AI

[Mistral](https://mistral.ai/) is a platform that provides a suite of AI models. Tabby supports Mistral's models for code completion and chat.

To connect Tabby with Mistral's models, you need to apply the following configurations in the `~/.tabby/config.toml` file:

```toml title="~/.tabby/config.toml"
# Completion Model
[model.completion.http]
kind = "mistral/completion"
model_name = "codestral-latest"
api_endpoint = "https://api.mistral.ai"
api_key = "secret-api-key"

# Chat Model
[model.chat.http]
kind = "mistral/chat"
model_name = "codestral-latest"
api_endpoint = "https://api.mistral.ai/v1"
api_key = "secret-api-key"
```
