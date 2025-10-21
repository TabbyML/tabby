use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tabby_inference::Embedding;

/// `MistralEmbeddingEngine` is responsible for interacting with Mistral's Embedding API.
#[derive(Clone)]
pub struct MistralEmbeddingEngine {
    client: Arc<Client>,
    api_endpoint: String,
    model_name: String,
    api_key: String,
}

/// Structure representing the request body for embedding.
#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    model: String,
    output_dtype: String,
    output_dimension: u32,
    input: Vec<String>,
}

/// Structure representing the response from the embedding API.
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<Data>,
}

/// Structure representing individual embedding data.
#[derive(Debug, Deserialize)]
struct Data {
    embedding: Vec<f32>,
}

impl MistralEmbeddingEngine {
    pub fn create(
        api_endpoint: Option<&str>,
        model_name: Option<&str>,
        api_key: Option<&str>,
    ) -> Box<dyn Embedding> {
        Box::new(Self {
            client: Arc::new(Client::new()),
            api_endpoint: format!(
                "{}/embeddings",
                api_endpoint.unwrap_or("https://api.mistral.ai/v1")
            ),
            model_name: model_name.unwrap_or("codestral-embed").to_owned(),
            api_key: api_key.unwrap_or_default().to_owned(),
        })
    }
}

#[async_trait]
impl Embedding for MistralEmbeddingEngine {
    async fn embed(&self, prompt: &str) -> Result<Vec<f32>> {
        // Clone all necessary fields to ensure thread safety across await points
        let request = EmbeddingRequest {
            model: self.model_name.clone(),
            output_dtype: "float".to_string(),
            // default as per https://docs.mistral.ai/capabilities/embeddings/code_embeddings/, max 3072
            output_dimension: 1536,
            input: vec![prompt.to_owned()],
        };

        // Send a POST request to the Mistral Embedding API
        let response = self
            .client
            .post(&self.api_endpoint)
            .bearer_auth(&self.api_key)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        // Check if the response status indicates success
        if !response.status().is_success() {
            let error_text = response.text().await?;
            anyhow::bail!("Mistral API error: {}", error_text);
        }

        // Deserialize the response body into `EmbeddingResponse`
        let embedding_response: EmbeddingResponse = response.json().await?;
        embedding_response
            .data
            .first()
            .map(|data| data.embedding.clone())
            .ok_or_else(|| anyhow::anyhow!("No embedding data received"))
    }
}
