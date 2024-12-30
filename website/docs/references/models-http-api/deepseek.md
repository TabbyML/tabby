# DeepSeek

[DeepSeek](https://www.deepseek.com/) offers a suite of AI models, such as [DeepSeek V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) and [DeepSeek Coder](https://huggingface.co/collections/deepseek-ai/deepseekcoder-v2-666bf4b274a5f556827ceeca), which perform well in coding tasks. Tabby supports DeepSeek's models for both code completion and chat.

Below is an example

```toml title="~/.tabby/config.toml"
# Chat model configuration
[model.chat.http]
# Deepseek's chat interface is compatible with OpenAI's chat API.
kind = "openai/chat"
model_name = "your_model"
api_endpoint = "https://api.deepseek.com/v1"
api_key = "secret-api-key"

# Completion model configuration
[model.completion.http]
# Deepseek uses its own completion API interface.
kind = "deepseek/completion"
model_name = "your_model"
api_endpoint = "https://api.deepseek.com/beta"
api_key = "secret-api-key"
```
