mod llama;

use std::sync::Arc;

use llama::LlamaCppEngine;
use tabby_common::config::HttpModelConfig;
use tabby_inference::Embedding;

pub fn create(config: &HttpModelConfig) -> Arc<dyn Embedding> {
    if config.kind == "llama.cpp/embedding" {
        let engine = LlamaCppEngine::create(&config.api_endpoint, config.api_key.clone());
        Arc::new(engine)
    } else {
        panic!("Only llama are supported for http embedding");
    }
}
