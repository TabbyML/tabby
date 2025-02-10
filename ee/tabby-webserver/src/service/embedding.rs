use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use tabby_common::config::EmbeddingConfig;
use tabby_inference::{clip_prompt, Embedding};

pub struct EmbeddingServiceImpl {
    config: EmbeddingConfig,
    embedding: Arc<dyn Embedding>,
}

pub fn create(config: &EmbeddingConfig, embedding: Arc<dyn Embedding>) -> Arc<dyn Embedding> {
    Arc::new(EmbeddingServiceImpl {
        config: config.clone(),
        embedding,
    })
}

#[async_trait]
impl Embedding for EmbeddingServiceImpl {
    async fn embed(&self, prompt: &str) -> Result<Vec<f32>> {
        if self.config.max_input_length != 0 && prompt.len() > self.config.max_input_length {
            self.embedding
                .embed(clip_prompt(prompt, self.config.max_input_length))
                .await
        } else {
            self.embedding.embed(prompt).await
        }
    }
}
