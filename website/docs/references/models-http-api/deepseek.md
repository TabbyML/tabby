# DeepSeek

[DeepSeek](https://www.deepseek.com/) is a platform that offers a suite of AI models. Tabby supports DeepSeek's models for both code completion and chat.

DeepSeek provides some OpenAI-compatible APIs, allowing us to use the OpenAI chat kinds directly.
However, for completion, there are some differences in the implementation, so we should use the `deepseek/completion` kind.

Below is an example

```toml title="~/.tabby/config.toml"
# Chat model
[model.chat.http]
kind = "openai/chat"
model_name = "your_model"
api_endpoint = "https://api.deepseek.com/v1"
api_key = "secret-api-key"

# Completion model
[model.completion.http]
kind = "deepseek/completion"
model_name = "your_model"
api_endpoint = "https://api.deepseek.com/beta"
api_key = "secret-api-key"
```
