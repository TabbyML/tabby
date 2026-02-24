# Azure OpenAI

[Azure OpenAI](https://azure.microsoft.com/products/ai-services/openai-service) is a cloud-based service that provides Azure customers with access to OpenAI's powerful language models including GPT-4, GPT-3.5, and various embedding models.

## Chat model

It supports various GPT series chat models through an Azure OpenAI-compatible API interface.

```toml title="~/.tabby/config.toml"
[[endpoints]]
name = "azure_chat"
api_route = "https://<resource-name>.openai.azure.com/openai/deployments/<deployment-id>"
headers = {
  Authorization = "Bearer your-api-key"
}
metadata = {
  pochi = {
    use_case = "chat",
    provider = "openai",
    models = [
      { name = "gpt-4o-mini", context_window = 128000 }
    ]
  }
}
```

## Completion model

Azure OpenAI currently does not offer completion-specific API endpoints.
