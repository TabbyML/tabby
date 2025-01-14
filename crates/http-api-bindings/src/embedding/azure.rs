use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tabby_inference::Embedding;

#[derive(Clone)]
pub struct AzureEmbeddingEngine {
    client: Arc<Client>,
    api_endpoint: String,
    api_key: String,
    api_version: String,
}

#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    input: String,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<Data>,
}

#[derive(Debug, Deserialize)]
struct Data {
    embedding: Vec<f32>,
}

impl AzureEmbeddingEngine {
    pub fn create(
        api_endpoint: &str,
        model_name: &str,
        api_key: Option<&str>,
        api_version: Option<&str>,
    ) -> Box<dyn Embedding> {
        let client = Client::new();
        let deployment_id = model_name;
        let azure_endpoint = format!(
            "{}/openai/deployments/{}/embeddings",
            api_endpoint.trim_end_matches('/'),
            deployment_id
        );

        Box::new(Self {
            client: Arc::new(client),
            api_endpoint: azure_endpoint,
            api_key: api_key.unwrap_or_default().to_owned(),
            api_version: api_version.unwrap_or("2024-02-15-preview").to_owned(),
        })
    }
}

#[async_trait]
impl Embedding for AzureEmbeddingEngine {
    async fn embed(&self, prompt: &str) -> Result<Vec<f32>> {
        let client = self.client.clone();
        let api_endpoint = self.api_endpoint.clone();
        let api_key = self.api_key.clone();
        let api_version = self.api_version.clone();
        let request = EmbeddingRequest {
            input: prompt.to_owned(),
        };

        let response = client
            .post(&api_endpoint)
            .query(&[("api-version", &api_version)])
            .header("api-key", &api_key)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            anyhow::bail!("Azure API error: {}", error_text);
        }

        let embedding_response: EmbeddingResponse = response.json().await?;
        embedding_response
            .data
            .first()
            .map(|data| data.embedding.clone())
            .ok_or_else(|| anyhow::anyhow!("No embedding data received"))
    }
}
