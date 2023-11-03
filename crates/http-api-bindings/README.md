## Usage

```bash
export MODEL_ID="code-gecko"
export PROJECT_ID="$(gcloud config get project)"
export API_ENDPOINT="https://us-central1-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/us-central1/publishers/google/models/${MODEL_ID}:predict"
export AUTHORIZATION="Bearer $(gcloud auth print-access-token)"
export TABBY_CONFIG=/the_dir_where_your_model_located/tabby.json

cargo run serve --device experimental-http --model "{\"kind\": \"vertex-ai\", \"api_endpoint\": \"$API_ENDPOINT\", \"authorization\": \"$AUTHORIZATION\", \"tabby_config\": \"$TABBY_CONFIG\"}"
```
