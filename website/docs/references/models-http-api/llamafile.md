# llamafile

[llamafile](https://github.com/Mozilla-Ocho/llamafile)
is a Mozilla Builders project that allows you to distribute and run LLMs with a single file.

llamafile provides an OpenAI API-compatible chat-completions and embedding endpoint,
enabling us to use the OpenAI kinds for chat and embeddings.

However, for completion, there are certain differences in the implementation, and we are still working on it.

llamafile uses port `8080` by default, which is also the port used by Tabby.
Therefore, it is recommended to run llamafile with the `--port` option to serve on a different port, such as `8081`.

Below is an example for chat:

```toml title="~/.tabby/config.toml"
# Chat model
[model.chat.http]
kind = "openai/chat"
model_name = "your_model"
api_endpoint = "http://localhost:8081/v1"
api_key = ""
```

For embeddings, the embedding endpoint is no longer supported in the standard llamafile server,
so you have to run llamafile with the `--embedding` option and set the Tabby config to:

```toml title="~/.tabby/config.toml"
# Embedding model
[model.embedding.http]
kind = "openai/embedding"
model_name = "your_model"
api_endpoint = "http://localhost:8082/v1"
api_key = ""
```