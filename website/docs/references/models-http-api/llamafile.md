# llamafile

[llamafile](https://github.com/Mozilla-Ocho/llamafile) is a Mozilla Builders project that allows you to distribute and run LLMs with a single file. It embeds a llama.cpp server and provides an OpenAI API-compatible chat-completions endpoint.

By default, llamafile uses port `8080`, which conflicts with Tabby's default port. It is recommended to run llamafile with the `--port` option to serve on a different port, such as `8081`. For embeddings functionality, you need to run llamafile with both the `--embedding` and `--port` options.

## Chat model

llamafile provides an OpenAI-compatible chat API interface. Note that the endpoint URL must include the `v1` suffix.

```toml title="~/.tabby/config.toml"
[[endpoints]]
name = "llamafile_chat"
api_route = "http://localhost:8081"
metadata = {
  pochi = {
    use_case = "chat",
    provider = "openai",
    models = [
      { name = "your_model", context_window = 4096 }
    ]
  }
}
```

<!-- FIXME(wei) update Completion config-->
## Completion model

llamafile uses llama.cpp's completion API interface. Note that the endpoint URL should NOT include the `v1` suffix.

```toml title="~/.tabby/config.toml"
[[endpoints]]
name = "llamafile_completion"
api_route = "http://localhost:8081"
metadata = {
  pochi = {
    use_case = "completion",
    provider = "openai",
    models = [
      { name = "your_model", context_window = 4096 }
    ]
  }
}
```
