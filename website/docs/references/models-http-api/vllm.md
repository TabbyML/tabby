# vLLM

[vLLM](https://docs.vllm.ai/en/stable/) is a fast and user-friendly library for LLM inference and serving. It provides an OpenAI-compatible server interface.

Important requirements for all model types:

- `model_name` must exactly match the one used to run vLLM
- `api_endpoint` should follow the format `http://host:port/v1`
- `api_key` should be identical to the one used to run vLLM

Please note that models differ in their capabilities for completion or chat. Some models can serve both purposes, please consult the model documentation for details.

## Chat model

vLLM provides an OpenAI-compatible chat API interface.

```toml title="~/.tabby/config.toml"
[[endpoints]]
name = "vllm_chat"
api_route = "http://localhost:8000"
headers = {
  Authorization = "Bearer your-api-key"
}
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

Due to implementation differences, vLLM uses its own completion API interface that requires a specific prompt template based on the model being used.

```toml title="~/.tabby/config.toml"
[model.completion.http]
kind = "vllm/completion"
model_name = "your_model"  # Please make sure to use a completion model
api_endpoint = "http://localhost:8000/v1"
api_key = "your-api-key"
prompt_template = "<PRE> {prefix} <SUF>{suffix} <MID>"  # Example prompt template for the CodeLlama model series
```
