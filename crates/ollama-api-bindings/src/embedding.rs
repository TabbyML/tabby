use async_trait::async_trait;
use ollama_rs::Ollama;
use tabby_common::config::HttpModelConfig;
use tabby_inference::Embedding;

use crate::model::OllamaModelExt;

pub struct OllamaCompletion {
    /// Connection to Ollama API
    connection: Ollama,
    /// Model name, <model>
    model: String,
}

#[async_trait]
impl Embedding for OllamaCompletion {
    async fn embed(&self, prompt: &str) -> anyhow::Result<Vec<f32>> {
        self.connection
            .generate_embeddings(self.model.to_owned(), prompt.to_owned(), None)
            .await
            .map(|x| x.embeddings)
            .map(|e| e.iter().map(|v| *v as f32).collect())
            .map_err(|err| err.into())
    }
}

pub async fn create(config: &HttpModelConfig) -> Box<dyn Embedding> {
    let connection = Ollama::try_new(config.api_endpoint.as_deref().unwrap().to_owned())
        .expect("Failed to create connection to Ollama, URL invalid");

    let model = connection.select_model_or_default(config).await.unwrap();

    Box::new(OllamaCompletion { connection, model })
}
