# Config.toml

Tabby uses `~/.tabby/config.toml` as configuration file by default,
you can adjust various aspects of its behavior, including:

- API Endpoint
- Model

:::info
Note that Tabby does not create this configuration file by default - you'll need to manually create the `config.toml` file in your `~/.tabby` directory.
:::

An example configuration file is shown below:

```toml
[[endpoints]]
name = "openai"
api_route = "https://api.openai.com"
timeout = 5000
headers = {
  Authorization = "Bearer TOKEN"
}
user_quota = {
  requests_per_minute = 1800
}
metadata = {
  pochi = {
    use_case = "chat",
    provider = "openai",
    models = [
      { name = "gpt-5", context_window = 200000 },
      { name = "gpt-5.2", context_window = 200000 }
    ]
  }
}

[model.chat.http]
kind = "openai/chat"
# Please make sure to use a chat model, such as gpt-4o
model_name = "gpt-4o"
# For multi-model support
supported_models = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]
api_endpoint = "https://api.openai.com/v1"
api_key = "your-api-key"

[model.completion.local]
model_id = "StarCoder2-3B"

[model.embedding.local]
model_id = "Nomic-Embed-Text"
```

## Endpoint configuration

You can configure Tabby to forward requests to external endpoints. For detailed configuration instructions, refer to [Endpoint Configuration](../endpoint).

## Model configuration

You can configure Tabby to connect to LLM models by setting up HTTP APIs. For detailed configuration instructions, refer to [Model Configuration](../model).
