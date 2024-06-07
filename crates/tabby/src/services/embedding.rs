use std::sync::Arc;

use tabby_common::config::ModelConfig;
use tabby_inference::Embedding;

use super::model;

pub async fn create(config: &ModelConfig) -> Arc<dyn Embedding> {
    model::load_embedding(config).await
}
