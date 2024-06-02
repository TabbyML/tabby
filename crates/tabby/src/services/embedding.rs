use std::sync::Arc;

use tabby_common::config::{LocalModelConfig, ModelConfig};
use tabby_inference::Embedding;

use super::model;

pub async fn create(config: Option<&ModelConfig>) -> Arc<dyn Embedding> {
    model::load_embedding(config.expect("Embedding model is not specified")).await
}
