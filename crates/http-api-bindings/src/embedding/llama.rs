use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tabby_inference::Embedding;
use tracing::Instrument;

use crate::{create_reqwest_client, embedding_info_span};

pub struct LlamaCppEngine {
    client: reqwest::Client,
    api_endpoint: String,
    api_key: Option<String>,
}

impl LlamaCppEngine {
    pub fn create(api_endpoint: &str, api_key: Option<String>) -> Box<dyn Embedding> {
        let client = create_reqwest_client(api_endpoint);

        Box::new(Self {
            client,
            api_endpoint: format!("{}/embedding", api_endpoint),
            api_key,
        })
    }
}

#[derive(Serialize)]
struct EmbeddingRequest {
    content: String,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    embedding: Vec<f32>,
}

#[async_trait]
impl Embedding for LlamaCppEngine {
    async fn embed(&self, prompt: &str) -> anyhow::Result<Vec<f32>> {
        let request = EmbeddingRequest {
            content: prompt.to_owned(),
        };

        let mut request = self.client.post(&self.api_endpoint).json(&request);
        if let Some(api_key) = &self.api_key {
            request = request.bearer_auth(api_key);
        }

        let response = request
            .send()
            .instrument(embedding_info_span!("llamacpp"))
            .await?;
        if response.status().is_server_error() {
            let error = response.text().await?;
            return Err(anyhow::anyhow!(
                "Error from server: {}, prompt length: {}",
                error,
                prompt.len()
            ));
        }

        let response = response.json::<EmbeddingResponse>().await?;
        Ok(response.embedding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This unit test should only run manually when the server is running
    /// curl -L https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf -o ./models/nomic.gguf
    /// ./server -m ./models/nomic.gguf --port 8000 --embedding
    #[tokio::test]
    #[ignore]
    async fn test_embedding() {
        let engine = LlamaCppEngine::create("http://localhost:8000", None);
        let embedding = engine.embed("hello").await.unwrap();
        assert_eq!(embedding.len(), 768);
    }
}
