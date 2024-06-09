# Model Configuration

Tabby incorporates two distinct model types: `Completion` and `Chat`. The `Completion` model is designed to provide suggestions for code completion, focusing mainly on the Fill-in-the-Middle (FIM) prompting style. On the other hand, the `Chat` model is adept at producing conversational replies and is broadly compatible with OpenAI's standards.

With the release of version 0.12, Tabby has rolled out an innovative model configuration system that facilitates linking Tabby to an HTTP API of a model. Furthermore, models listed in the [Model Registry](/docs/models) may be set up as a `local` backend. In this arrangement, Tabby initiates the `llama-server` as a subprocess and seamlessly establishes a connection to the model via the subprocess's HTTP API.

### Completion Model

#### [local](/docs/models)

To configure the `local` model, use the following settings:

```toml
[model.completion.local]
model_id = "StarCoder2-3B"
```

#### [llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md#api-endpoints)

The `llama.cpp` model can be configured with the following parameters:

```toml
[model.completion.http]
kind = "llama.cpp/completion"
api_endpoint = "http://localhost:8888"
```

#### [ollama](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion)

For setting up the `ollama` model, apply the configuration below:

```toml
[model.completion.http]
kind = "ollama/completion"
api_endpoint = "http://localhost:8888"
```

#### [mistral / codestral](https://docs.mistral.ai/api/#operation/createFIMCompletion)

Configure the `mistral/codestral` model as follows:

```toml
[model.completion.http]
kind = "mistral/completion"
api_endpoint = "https://api.mistral.ai"
api_key = "secret-api-key"
```

### Chat Model

Chat models adhere to the standard interface specified by OpenAI's `/chat/completions` API.


#### local

For `local` configuration, use:

```toml
[model.chat.local]
model_id = "StarCoder2-3B"
```

#### http

For `HTTP` configuration, the settings are as follows:

```toml
[model.chat.http]
kind = "openai/chat"
model_name = "deepseek-chat"
api_endpoint = "https://api.deepseek.com/v1"
api_key = "secret-api-key"
```
