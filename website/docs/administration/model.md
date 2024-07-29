# Model Configuration

You can configure how Tabby connect with LLM models by editing the `~/.tabby/config.toml` file. Tabby incorporates two distinct model types: `Completion` and `Chat`. The `Completion` model is designed to provide suggestions for code completion, focusing mainly on the Fill-in-the-Middle (FIM) prompting style. On the other hand, the `Chat` model is adept at producing conversational replies and is broadly compatible with OpenAI's standards.

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
prompt_template = "<PRE> {prefix} <SUF>{suffix} <MID>"  # Example prompt template for CodeLlama model series.
```

#### [ollama](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion)

For setting up the `ollama` model, apply the configuration below:

```toml
[model.completion.http]
kind = "ollama/completion"
model_name = "codellama:7b"
api_endpoint = "http://localhost:8888"
prompt_template = "<PRE> {prefix} <SUF>{suffix} <MID>"  # Example prompt template for CodeLlama model series.
```

#### [mistral / codestral](https://docs.mistral.ai/api/#operation/createFIMCompletion)

Configure the `mistral/codestral` model as follows:

```toml
[model.completion.http]
kind = "mistral/completion"
api_endpoint = "https://api.mistral.ai"
api_key = "secret-api-key"
```

#### [openai completion](https://platform.openai.com/docs/api-reference/completions)

Configure Tabby with an OpenAI-compatible completion model (`/v1/completions`) using an online service or a self-hosted backend (vLLM, Nvidia NIM, LocalAI, ...) as follows:

```toml
[model.completion.http]
kind = "openai/completion"
model_name = "your_model"
api_endpoint = "https://url_to_your_backend_or_service"
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

#### openai/chat

To configure Tabby's chat functionality with an OpenAI-compatible chat model (`/v1/chat/completions`), apply the settings below. This example uses the API platform of DeepSeek. Similar configurations can be applied for other LLM vendors such as Mistral, OpenAI, etc.

```toml
[model.chat.http]
kind = "openai/chat"
model_name = "deepseek-chat"
api_endpoint = "https://api.deepseek.com/v1"
api_key = "secret-api-key"
```

#### [mistral / codestral](https://docs.mistral.ai/api/#operation/createFIMCompletion)

Configure the `mistral/codestral` model as follows:

```toml
[model.completion.http]
kind = "mistral/chat"
api_endpoint = "https://api.mistral.ai"
api_key = "secret-api-key"
```

### Embedding Model

Tabby utilize embedding models to convert documents and queries into vectors for efficient context retrieval. The default embedding model is `Nomic-Embed-Text`, which is a high-performing open embedding model with a large token context window. Currently, `Nomic-Embed-Text` is the only supported local embedding model.

### Using a remote embedding model provider

You can add also a remote embedding model provider by adding a new section to the `~/.tabby/config.toml` file.

```toml
[model.embedding.http]
kind = "openai/embedding"
api_endpoint = "https://api.openai.com"
api_key = "sk-..."
model_name = "text-embedding-3-small"
```

Following embedding model providers are supported:

* `openai/embedding`
* `voyageai/embedding`
* `llama.cpp/embedding`
* `ollama/embedding`