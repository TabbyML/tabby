## Examples

## Vertex.AI

```bash
export MODEL_ID="code-gecko"
export PROJECT_ID="$(gcloud config get project)"
export API_ENDPOINT="https://us-central1-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/us-central1/publishers/google/models/${MODEL_ID}:predict"
export AUTHORIZATION="Bearer $(gcloud auth print-access-token)"

cargo run serve --device experimental-http --model "{\"kind\": \"vertex-ai\", \"api_endpoint\": \"$API_ENDPOINT\", \"authorization\": \"$AUTHORIZATION\"}"
```

### OpenAI

```bash
cargo run serve --device experimental-http \
  --model '{"kind": "openai", "model_name": "codellama/CodeLlama-70b-Instruct-hf", "api_endpoint": "http://host/v1/completions", "prompt_template": "{prefix}"}'
```