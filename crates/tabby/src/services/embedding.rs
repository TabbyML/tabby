use std::sync::Arc;

use tabby_common::config::{LocalModelConfig, ModelConfig};
use tabby_inference::Embedding;

use super::model;

pub async fn create(config: Option<&ModelConfig>) -> Arc<dyn Embedding> {
    if let Some(config) = config {
        model::load_embedding(config).await
    } else {
        model::load_embedding(&default_config()).await
    }
}

fn default_config() -> ModelConfig {
    ModelConfig::Local(LocalModelConfig {
        model_id: "Nomic-Embed-Text".to_string(),
        parallelism: 4,
        num_gpu_layers: 9999,
    })
}
