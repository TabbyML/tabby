# vLLM

[vLLM](https://docs.vllm.ai/en/stable/) is a fast and user-friendly library for LLM inference and serving.

vLLM offers an `OpenAI Compatible Server`, enabling us to use the OpenAI kinds for chat and embedding.
However, for completion, there are certain differences in the implementation.
Therefore, we should use the `vllm/completion` kind and provide a `prompt_template` depending on the specific models.

Please note that models differ in their capabilities for completion or chat.
You should confirm the model's capability before employing it for chat or completion tasks.

Additionally, there are models that can serve both as chat and completion.
For detailed information, please refer to the [Model Registry](../../models/index.mdx).

Below is an example of the vLLM running at `http://localhost:8000`:

Please note the following requirements in each model type:
1. `model_name` must exactly match the one used to run vLLM.
2. `api_endpoint` should follow the format `http://host:port/v1`.
3. `api_key` should be identical to the one used to run vLLM.

```toml title="~/.tabby/config.toml"
# Chat model
[model.chat.http]
kind = "openai/chat"
model_name = "your_model"   # Please make sure to use a chat model.
api_endpoint = "http://localhost:8000/v1"
api_key = "secret-api-key"

# Completion model
[model.completion.http]
kind = "vllm/completion"
model_name = "your_model"  # Please make sure to use a completion model.
api_endpoint = "http://localhost:8000/v1"
api_key = "secret-api-key"
prompt_template = "<PRE> {prefix} <SUF>{suffix} <MID>"  # Example prompt template for the CodeLlama model series.

# Embedding model
[model.embedding.http]
kind = "openai/embedding"
model_name = "your_model"
api_endpoint = "http://localhost:8000/v1"
api_key = "secret-api-key"
```
