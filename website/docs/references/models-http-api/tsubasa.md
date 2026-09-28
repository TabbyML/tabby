# Tsubasa

Use Tsubasa through Tabby's existing OpenAI-compatible chat connector.

## Chat model

```toml title="~/.tabby/config.toml"
[model.chat.http]
kind = "openai/chat"
model_name = "tsubasa-fast"
supported_models = ["tsubasa-fast", "tsubasa-pro"]
api_endpoint = "https://api.tsubasa.sh/v1"
api_key = "your-tsubasa-api-key"
```

Keep the API key in your private Tabby configuration. Do not append
`/chat/completions` to `api_endpoint`. Set `model_name` to `tsubasa-pro` to use
Pro as the default; both aliases can also be selected through `supported_models`.

Both models have a 32,768-token context window shared by input and output. Fast
allows up to 8,192 output tokens; Pro allows up to 16,384. Keep enough room for
output when sending conversation history or retrieved context.

This configuration covers text chat. Configure completion/FIM and embedding
models separately if you need those features.
