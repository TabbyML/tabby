use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tabby_inference::Embedding;

/// `AzureEmbeddingEngine` is responsible for interacting with Azure's Embedding API.
///
/// **Note**: Currently, this implementation only supports the OpenAI API and specific API versions.
#[derive(Clone)]
pub struct CodestralEmbeddingEngine {
    client: Arc<Client>,
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

impl CodestralEmbeddingEngine {
    pub fn create(api_key: Option<&str>) -> Box<dyn Embedding> {
        Box::new(Self {
            client: Arc::new(Client::new()),
            api_key: api_key.unwrap_or_default().to_owned(),
        })
    }
}

#[async_trait]
impl Embedding for CodestralEmbeddingEngine {
    async fn embed(&self, prompt: &str) -> Result<Vec<f32>> {
        // Clone all necessary fields to ensure thread safety across await points
        let api_endpoint = "https://api.mistral.ai/v1/embeddings".to_string();
        let request = EmbeddingRequest {
            model: "codestral-embed".to_string(),
            output_dtype: "float".to_string(),
            // default as per https://docs.mistral.ai/capabilities/embeddings/code_embeddings/, max 3072
            output_dimension: 1536,
            input: vec![prompt.to_owned()],
        };

        // Send a POST request to the Azure Embedding API
        let response = self
            .client
            .post(&api_endpoint)
            .bearer_auth(&self.api_key)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        // Check if the response status indicates success
        if !response.status().is_success() {
            let error_text = response.text().await?;
            anyhow::bail!("Codestral API error: {}", error_text);
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
