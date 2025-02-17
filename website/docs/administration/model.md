# Model Configuration

You can configure how Tabby connects with LLM models by editing the `~/.tabby/config.toml` file. Tabby incorporates three types of models: **Completion**, **Chat**, and **Embedding**. Each of them can be configured individually.

- **Completion Model**: The Completion model is designed to provide suggestions for code completion, focusing mainly on the Fill-in-the-Middle (FIM) prompting style.
- **Chat Model**: The Chat model is adept at producing conversational replies and is broadly compatible with OpenAI's standards.
- **Embedding Model**: The Embedding model is used to generate embeddings for text data, by default Tabby uses the `Nomic-Embed-Text` model.

Each of the model types can be configured with either a local model or a remote model provider. For local models, Tabby will initiate a subprocess (powered by [llama.cpp](https://github.com/ggml-org/llama.cpp)) and connect to the model via an HTTP API. For remote models, Tabby will connect directly to the model provider's API.

Below is an example of how to configure the model settings in the `~/.tabby/config.toml` file:

```toml
[model.completion.local]
model_id = "StarCoder2-3B"

[model.chat.local]
model_id = "Mistral-7B"

[model.embedding.local]
model_id = "Nomic-Embed-Text"
```

More supported models can be found in the [Model Registry](../../models). For configuring model through HTTP API, check [References / Models HTTP API](../../references/models-http-api/llama.cpp).
