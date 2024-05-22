mod llama;
mod openai;

use std::sync::Arc;

use llama::LlamaCppEngine;
use tabby_common::config::HttpModelConfig;
use tabby_inference::Embedding;

use self::openai::OpenAIEmbeddingEngine;

pub fn create(config: &HttpModelConfig) -> Arc<dyn Embedding> {
    if config.kind == "llama.cpp/embedding" {
        let engine = LlamaCppEngine::create(&config.api_endpoint, config.api_key.clone());
        Arc::new(engine)
    } else if config.kind == "openai-embedding" {
        let engine = OpenAIEmbeddingEngine::create(
            &config.api_endpoint,
            config.model_name.as_deref().unwrap_or_default(),
            config.api_key.clone(),
        );
        Arc::new(engine)
    } else {
        panic!("Only llama are supported for http embedding");
    }
}
