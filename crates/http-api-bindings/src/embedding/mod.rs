mod llama;
mod openai;
mod rate_limit;
mod voyage;

use core::panic;
use std::{sync::Arc, time::Duration};

use llama::LlamaCppEngine;
use ratelimit::Ratelimiter;
use tabby_common::config::HttpModelConfig;
use tabby_inference::Embedding;

use self::{openai::OpenAIEmbeddingEngine, voyage::VoyageEmbeddingEngine};

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

    let ratelimiter = Ratelimiter::builder(
        config.rate_limit.request_per_minute,
        Duration::from_secs(60),
    )
    .max_tokens(config.rate_limit.request_per_minute)
    .build()
    .expect("Failed to create ratelimiter, please check the rate limit configuration");

    Arc::new(rate_limit::RateLimitedEmbedding::new(engine, ratelimiter))
}
