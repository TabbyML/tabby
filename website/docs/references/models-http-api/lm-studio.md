# LM Studio

[LM Studio](https://lmstudio.ai/) is a desktop application that allows you to discover, download, and run local LLMs using various model formats (GGUF, GGML, SafeTensors). It provides an OpenAI-compatible API server for running these models locally.

## Chat model

LM Studio provides an OpenAI-compatible chat API interface that can be used with Tabby.

```toml title="~/.tabby/config.toml"
[model.chat.http]
kind = "openai/chat"
model_name = "deepseek-r1-distill-qwen-7b"  # Example model
api_endpoint = "http://localhost:1234/v1"    # LM Studio server endpoint with /v1 path
api_key = ""                                 # No API key required for local deployment
```

## Completion model

LM Studio can be used for code completion tasks through its OpenAI-compatible completion API.

```toml title="~/.tabby/config.toml"
[model.completion.http]
kind = "openai/completion"
model_name = "starcoder2-7b"                 # Example code completion model
api_endpoint = "http://localhost:1234/v1"
api_key = ""
prompt_template = "<PRE> {prefix} <SUF>{suffix} <MID>"  # Example prompt template for CodeLlama models
```

## Embeddings model

LM Studio supports embedding functionality through its OpenAI-compatible API.

```toml title="~/.tabby/config.toml"
[model.embedding.http]
kind = "openai/embedding"
model_name = "text-embedding-nomic-embed-text-v1.5"
api_endpoint = "http://localhost:1234/v1"
api_key = ""
```

## Usage Notes

1. Download and install LM Studio from their [official website](https://lmstudio.ai/).
2. Download your preferred model through LM Studio's model discovery interface.
3. Start the local server by clicking the "Start Server" button in LM Studio.
4. Configure Tabby to use LM Studio's API endpoint as shown in the examples above.
5. The default server port is 1234, but you can change it in LM Studio's settings if needed.
6. Make sure to append `/v1` to the API endpoint as LM Studio follows OpenAI's API structure.

LM Studio is particularly useful for running models locally without requiring complex setup or command-line knowledge. It supports a wide range of models and provides a user-friendly interface for model management and server operations.
