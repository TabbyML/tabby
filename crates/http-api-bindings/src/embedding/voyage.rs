use anyhow::Context;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tabby_inference::Embedding;

const DEFAULT_VOYAGE_API_ENDPOINT: &str = "https://api.voyageai.com";

pub struct VoyageEmbeddingEngine {
    client: Client,
    api_endpoint: String,
    api_key: String,
    model_name: String,
}

impl VoyageEmbeddingEngine {
    pub fn create(api_endpoint: &str, model_name: &str, api_key: String) -> Self {
        let endpoint = if api_endpoint.is_empty() {
            DEFAULT_VOYAGE_API_ENDPOINT
        } else {
            api_endpoint
        };
        let client = Client::new();
        Self {
            client,
            api_endpoint: format!("{}/v1/embeddings", endpoint),
            api_key,
            model_name: model_name.to_owned(),
        }
    }
}

#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    input: Vec<String>,
    model: String,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

#[async_trait]
impl Embedding for VoyageEmbeddingEngine {
    async fn embed(&self, prompt: &str) -> anyhow::Result<Vec<f32>> {
        let request = EmbeddingRequest {
            input: vec![prompt.to_owned()],
            model: self.model_name.clone(),
        };

        let request_builder = self
            .client
            .post(&self.api_endpoint)
            .json(&request)
            .header("content-type", "application/json")
            .bearer_auth(&self.api_key);

        let response = request_builder.send().await?;
        if response.status().is_server_error() {
            let error = response.text().await?;
            return Err(anyhow::anyhow!("Error from server: {}", error));
        }

        let response_body = response
            .json::<EmbeddingResponse>()
            .await
            .context("Failed to parse response body")?;

        response_body
            .data
            .into_iter()
            .next()
            .map(|data| data.embedding)
            .ok_or_else(|| anyhow::anyhow!("No embedding data found"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Make sure you have set the VOYAGE_API_KEY environment variable before running the test
    #[tokio::test]
    #[ignore]
    async fn test_voyage_embedding() {
        let api_key = std::env::var("VOYAGE_API_KEY").expect("VOYAGE_API_KEY must be set");
        let engine =
            VoyageEmbeddingEngine::create(DEFAULT_VOYAGE_API_ENDPOINT, "voyage-code-2", api_key);
        let embedding = engine.embed("Hello, world!").await.unwrap();
        assert_eq!(embedding.len(), 1536);
    }
}
