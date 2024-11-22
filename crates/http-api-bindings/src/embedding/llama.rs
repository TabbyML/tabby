use std::time;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tabby_inference::Embedding;

use crate::RateLimitedClient;

pub struct LlamaCppEngine {
    client: RateLimitedClient,
    api_endpoint: String,
}

impl LlamaCppEngine {
    pub fn create(api_endpoint: &str, api_key: Option<String>, num_request: u64, per: u64) -> Self {
        let client = RateLimitedClient::new(
            api_endpoint,
            api_key,
            num_request,
            time::Duration::from_secs(per),
        )
        .unwrap();

        Self {
            client,
            api_endpoint: format!("{}/embedding", api_endpoint),
        }
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

        let response = self.client.post(&self.api_endpoint, request).await?;
        if response.status().is_server_error() {
            let error = response.text().await?;
            return Err(anyhow::anyhow!(
                "Error from server: {}, prompt: {}",
                error,
                prompt
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
    async fn test_embedding_no_limit() {
        let engine = LlamaCppEngine::create("http://localhost:8000", None, 0, 0);
        let embedding = engine.embed("hello").await.unwrap();
        assert_eq!(embedding.len(), 768);
    }
}
