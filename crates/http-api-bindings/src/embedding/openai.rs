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
    pub fn create(api_endpoint: &str, model_name: &str, api_key: Option<&str>) -> impl Embedding {
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
}
