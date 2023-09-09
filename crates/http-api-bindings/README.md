## Usage

```bash
MODEL_ID="code-gecko"
PROJECT_ID=PROJECT_ID
API_ENDPOINT="https://us-central1-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/us-central1/publishers/google/models/${MODEL_ID}:predict"
AUTHORIZATION="Bearer $(gcloud auth print-access-token)"

cargo run --example simple
```
