## Usage

```bash
export MODEL_ID="code-gecko"
export PROJECT_ID=PROJECT_ID
export API_ENDPOINT="https://us-central1-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/us-central1/publishers/google/models/${MODEL_ID}:predict"
export AUTHORIZATION="Bearer $(gcloud auth print-access-token)"

cargo run --example simple
```
