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
    pub fn create(
        api_endpoint: Option<&str>,
        model_name: &str,
        api_key: String,
    ) -> Box<dyn Embedding> {
        let api_endpoint = api_endpoint.unwrap_or(DEFAULT_VOYAGE_API_ENDPOINT);
        let client = Client::new();
        Box::new(Self {
            client,
            api_endpoint: format!("{}/v1/embeddings", api_endpoint),
            api_key,
            model_name: model_name.to_owned(),
        })
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
        if !response.status().is_success() {
            let status = response.status();
            let error = response.text().await?;
            return Err(anyhow::anyhow!("Error {}: {}", status.as_u16(), error));
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

    /// VOYAGE_API_KEY=xxx cargo test test_voyage_embedding -- --ignored
    #[tokio::test]
    #[ignore]
    async fn test_voyage_embedding() {
        let api_key = std::env::var("VOYAGE_API_KEY").expect("VOYAGE_API_KEY must be set");
        let engine = VoyageEmbeddingEngine::create(None, "voyage-code-2", api_key);
        let embedding = engine.embed("Hello, world!").await.unwrap();
        assert_eq!(embedding.len(), 1536);
    }
}
