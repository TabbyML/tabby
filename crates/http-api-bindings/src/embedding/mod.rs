mod llama;
mod openai;
mod voyage;

use core::panic;
use std::sync::Arc;

use llama::LlamaCppEngine;
use tabby_common::config::HttpModelConfig;
use tabby_inference::Embedding;

use self::{openai::OpenAIEmbeddingEngine, voyage::VoyageEmbeddingEngine};
use super::rate_limit;

pub async fn create(config: &HttpModelConfig) -> Arc<dyn Embedding> {
    let engine = match config.kind.as_str() {
        "llama.cpp/embedding" => LlamaCppEngine::create(
            config
                .api_endpoint
                .as_deref()
                .expect("api_endpoint is required"),
            config.api_key.clone(),
        ),
        "openai/embedding" => OpenAIEmbeddingEngine::create(
            config
                .api_endpoint
                .as_deref()
                .expect("api_endpoint is required"),
            config.model_name.as_deref().unwrap_or_default(),
            config.api_key.as_deref(),
        ),
        "ollama/embedding" => ollama_api_bindings::create_embedding(config).await,
        "voyage/embedding" => VoyageEmbeddingEngine::create(
            config.api_endpoint.as_deref(),
            config
                .model_name
                .as_deref()
                .expect("model_name must be set for voyage/embedding"),
            config
                .api_key
                .clone()
                .expect("api_key must be set for voyage/embedding"),
        ),
        unsupported_kind => panic!(
            "Unsupported kind for http embedding model: {}",
            unsupported_kind
        ),
    };

    Arc::new(rate_limit::new_embedding(
        engine,
        config.rate_limit.request_per_minute,
    ))
}
