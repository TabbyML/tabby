use anyhow::Context;
use async_openai::{
    config::OpenAIConfig,
    types::{CreateEmbeddingRequest, EmbeddingInput},
};
use async_trait::async_trait;
use tabby_inference::Embedding;

pub struct OpenAIEmbeddingEngine {
    client: async_openai::Client<OpenAIConfig>,
    model_name: String,
}

impl OpenAIEmbeddingEngine {
    pub fn create(api_endpoint: &str, model_name: &str, api_key: Option<String>) -> Self {
        let config = OpenAIConfig::default()
            .with_api_base(api_endpoint)
            .with_api_key(api_key.unwrap_or_default());

        let client = async_openai::Client::with_config(config);

        Self {
            client,
            model_name: model_name.to_owned(),
        }
    }
}

#[async_trait]
impl Embedding for OpenAIEmbeddingEngine {
    async fn embed(&self, prompt: &str) -> anyhow::Result<Vec<f32>> {
        let request = CreateEmbeddingRequest {
            model: self.model_name.clone(),
            input: EmbeddingInput::String(prompt.to_owned()),
            encoding_format: None,
            user: None,
            dimensions: None,
        };
        let resp = self.client.embeddings().create(request).await?;
        let data = resp
            .data
            .into_iter()
            .next()
            .context("Failed to get embedding")?;
        Ok(data.embedding)
    }
}
