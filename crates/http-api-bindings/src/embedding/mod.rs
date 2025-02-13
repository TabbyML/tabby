mod azure;
mod llama;
mod openai;

use core::panic;
use std::sync::Arc;

use azure::AzureEmbeddingEngine;
use llama::LlamaCppEngine;
use openai::OpenAIEmbeddingEngine;
use tabby_common::config::HttpModelConfig;
use tabby_inference::Embedding;

use super::rate_limit;

pub async fn create(config: &HttpModelConfig) -> Arc<dyn Embedding> {
    let engine = match config.kind.as_str() {
        "llama.cpp/embedding" => LlamaCppEngine::create(
            config
                .api_endpoint
                .as_deref()
                .expect("api_endpoint is required"),
            config.api_key.clone(),
            false,
        ),
        "llama.cpp/before_b4356_embedding" => LlamaCppEngine::create(
            config
                .api_endpoint
                .as_deref()
                .expect("api_endpoint is required"),
            config.api_key.clone(),
            true,
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
        "voyage/embedding" => OpenAIEmbeddingEngine::create(
            config
                .api_endpoint
                .as_deref()
                .unwrap_or("https://api.voyageai.com/v1"),
            config
                .model_name
                .as_deref()
                .expect("model_name must be set for voyage/embedding"),
            config.api_key.as_deref(),
        ),
        "azure/embedding" => AzureEmbeddingEngine::create(
            config
                .api_endpoint
                .as_deref()
                .expect("api_endpoint is required for azure/embedding"),
            config.model_name.as_deref().unwrap_or_default(), // Provide a default if model_name is optional
            config.api_key.as_deref(),
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

#[macro_export]
macro_rules! embedding_info_span {
    ($kind:expr) => {
        tracing::info_span!("embedding", kind = $kind)
    };
}
