use anyhow::Context;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tabby_inference::Embedding;
use tracing::Instrument;

use crate::embedding_info_span;

pub struct OpenAIEmbeddingEngine {
    client: Client,
    api_endpoint: String,
    api_key: String,
    model_name: String,
}

impl OpenAIEmbeddingEngine {
    pub fn create(
        api_endpoint: &str,
        model_name: &str,
        api_key: Option<&str>,
    ) -> Box<dyn Embedding> {
        let client = Client::new();
        Box::new(Self {
            client,
            api_endpoint: format!("{api_endpoint}/embeddings"),
            api_key: api_key.unwrap_or_default().to_owned(),
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
impl Embedding for OpenAIEmbeddingEngine {
    async fn embed(&self, prompt: &str) -> anyhow::Result<Vec<f32>> {
        let request = EmbeddingRequest {
            input: vec![prompt.to_owned()],
            model: self.model_name.clone(),
        };

        let request_builder = self
            .client
            .post(&self.api_endpoint)
            .json(&request)
            .bearer_auth(&self.api_key);

        let response = request_builder
            .send()
            .instrument(embedding_info_span!("openai"))
            .await?;

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

    /// Make sure you have set the JINA_API_KEY environment variable before running the test
    #[tokio::test]
    #[ignore]
    async fn test_jina_embedding() {
        let api_key = std::env::var("JINA_API_KEY").expect("JINA_API_KEY must be set");
        let engine = OpenAIEmbeddingEngine::create(
            "https://api.jina.ai/v1",
            "jina-embeddings-v2-base-en",
            Some(&api_key),
        );
        let embedding = engine.embed("Hello, world!").await.unwrap();
        assert_eq!(embedding.len(), 768);
    }

    #[tokio::test]
    #[ignore]
    async fn test_voyage_embedding() {
        let api_key = std::env::var("VOYAGE_API_KEY").expect("VOYAGE_API_KEY must be set");
        let engine = OpenAIEmbeddingEngine::create(
            "https://api.voyageai.com/v1",
            "voyage-code-2",
            Some(&api_key),
        );
        let embedding = engine.embed("Hello, world!").await.unwrap();
        assert_eq!(embedding.len(), 1536);
    }
}
