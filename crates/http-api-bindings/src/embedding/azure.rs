use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tabby_inference::Embedding;

use crate::AZURE_API_VERSION;

/// `AzureEmbeddingEngine` is responsible for interacting with Azure's Embedding API.
///
/// **Note**: Currently, this implementation only supports the OpenAI API and specific API versions.
#[derive(Clone)]
pub struct AzureEmbeddingEngine {
    client: Arc<Client>,
    api_endpoint: String,
    api_key: String,
}

/// Structure representing the request body for embedding.
#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    input: String,
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

impl AzureEmbeddingEngine {
    /// Creates a new instance of `AzureEmbeddingEngine`.
    ///
    /// **Note**: Currently, this implementation only supports the OpenAI API and specific API versions.
    ///
    /// # Parameters
    ///
    /// - `api_endpoint`: The base URL of the Azure Embedding API.
    /// - `model_name`: The name of the deployed model, used to construct the deployment ID.
    /// - `api_key`: Optional API key for authentication.
    /// - `api_version`: Optional API version, defaults to "2023-05-15".
    ///
    /// # Returns
    ///
    /// A boxed instance that implements the `Embedding` trait.
    pub fn create(
        api_endpoint: &str,
        model_name: &str,
        api_key: Option<&str>,
    ) -> Box<dyn Embedding> {
        let client = Client::new();
        let deployment_id = model_name;
        // Construct the full endpoint URL for the Azure Embedding API
        let azure_endpoint = format!(
            "{}/openai/deployments/{}/embeddings",
            api_endpoint.trim_end_matches('/'),
            deployment_id
        );

        Box::new(Self {
            client: Arc::new(client),
            api_endpoint: azure_endpoint,
            api_key: api_key.unwrap_or_default().to_owned(),
        })
    }
}

#[async_trait]
impl Embedding for AzureEmbeddingEngine {
    /// Generates an embedding vector for the given prompt.
    ///
    /// **Note**: Currently, this implementation only supports the OpenAI API and specific API versions.
    ///
    /// # Parameters
    ///
    /// - `prompt`: The input text to generate embeddings for.
    ///
    /// # Returns
    ///
    /// A `Result` containing the embedding vector or an error.
    async fn embed(&self, prompt: &str) -> Result<Vec<f32>> {
        // Clone all necessary fields to ensure thread safety across await points
        let api_endpoint = self.api_endpoint.clone();
        let api_key = self.api_key.clone();
        let api_version = AZURE_API_VERSION.to_string();
        let request = EmbeddingRequest {
            input: prompt.to_owned(),
        };

        // Send a POST request to the Azure Embedding API
        let response = self
            .client
            .post(&api_endpoint)
            .query(&[("api-version", &api_version)])
            .header("api-key", &api_key)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        // Check if the response status indicates success
        if !response.status().is_success() {
            let error_text = response.text().await?;
            anyhow::bail!("Azure API error: {}", error_text);
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
