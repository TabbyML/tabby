# LM Studio

[LM Studio](https://lmstudio.ai/) is a desktop application that allows you to discover, download, and run local LLMs using various model formats (GGUF, GGML, SafeTensors). It provides an OpenAI-compatible API server for running these models locally.

## Chat model

LM Studio provides an OpenAI-compatible chat API interface that can be used with Tabby.

```toml title="~/.tabby/config.toml"
[[endpoints]]
name = "lmstudio_chat"
api_route = "http://localhost:1234"
metadata = {
  pochi = {
    use_case = "chat",
    provider = "openai",
    models = [
      { name = "deepseek-r1-distill-qwen-7b", context_window = 32000 }
    ]
  }
}
```

<!-- FIXME(wei) update Completion config-->
## Completion model

LM Studio can be used for code completion tasks through its OpenAI-compatible completion API.

```toml title="~/.tabby/config.toml"
[[endpoints]]
name = "lmstudio_completion"
api_route = "http://localhost:1234/v1"
metadata = {
  pochi = {
    use_case = "completion",
    provider = "openai",
    models = [
      { name = "starcoder2-7b", context_window = 16000 }
    ]
  }
}
```

## Usage Notes

1. Download and install LM Studio from their [official website](https://lmstudio.ai/).
2. Download your preferred model through LM Studio's model discovery interface.
3. Start the local server by clicking the "Start Server" button in LM Studio.
4. Configure Tabby to use LM Studio's API endpoint as shown in the examples above.
5. The default server port is 1234, but you can change it in LM Studio's settings if needed.

LM Studio is particularly useful for running models locally without requiring complex setup or command-line knowledge. It supports a wide range of models and provides a user-friendly interface for model management and server operations.
