mod llama;
mod openai;

use core::panic;
use std::sync::Arc;

use llama::LlamaCppEngine;
use tabby_common::config::HttpModelConfig;
use tabby_inference::Embedding;

use self::openai::OpenAIEmbeddingEngine;

pub async fn create(config: &HttpModelConfig) -> Arc<dyn Embedding> {
    match config.kind.as_str() {
        "llama.cpp/embedding" => {
            let engine = LlamaCppEngine::create(&config.api_endpoint, config.api_key.clone());
            Arc::new(engine)
        }
        "openai-embedding" => {
            let engine = OpenAIEmbeddingEngine::create(
                &config.api_endpoint,
                config.model_name.as_deref().unwrap_or_default(),
                config.api_key.clone(),
            );
            Arc::new(engine)
        }
        "ollama/embedding" => ollama_api_bindings::create_embedding(config).await,

        unsupported_kind => panic!(
            "Unsupported kind for http embedding model: {}",
            unsupported_kind
        ),
    }
}
