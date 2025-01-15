# Perplexity AI

[Perplexity AI](https://www.perplexity.ai/) is a company that develops large language models and offers them through their API service. They currently provide three powerful Llama-based models: [Sonar Small (8B)](https://docs.perplexity.ai/guides/model-cards#supported-models), [Sonar Large (70B)](https://docs.perplexity.ai/guides/model-cards#supported-models), and [Sonar Huge (405B)](https://docs.perplexity.ai/guides/model-cards#supported-models), all supporting a 128k context window.

## Chat model

Perplexity provides an OpenAI-compatible chat API interface. The Sonar Large (70B) and Huge (405B) models are recommended for better performance.

```toml title="~/.tabby/config.toml"
[model.chat.http]
kind = "openai/chat"
model_name = "llama-3.1-sonar-large-128k-online"  # Also supports sonar-small-128k-online or sonar-huge-128k-online
api_endpoint = "https://api.perplexity.ai"
api_key = "your-api-key"
```

## Completion model

Perplexity currently does not offer completion-specific API endpoints.

## Embeddings model

Perplexity currently does not offer embeddings models through their API.
