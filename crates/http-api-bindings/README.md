## Examples

### OpenAI

```bash
cargo run serve --device experimental-http \
  --model '{"kind": "openai", "model_name": "codellama/CodeLlama-70b-Instruct-hf", "api_endpoint": "http://host/v1/completions", "prompt_template": "{prefix}"}'
```