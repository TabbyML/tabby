mod llama;

use std::sync::Arc;

use llama::LlamaCppEngine;
use tabby_common::config::HttpModelConfig;
use tabby_inference::CompletionStream;

pub fn create(model: &HttpModelConfig) -> Arc<dyn CompletionStream> {
    if model.kind == "llama.cpp/completion" {
        let engine = LlamaCppEngine::create(&model.api_endpoint, model.api_key.clone());
        Arc::new(engine)
    } else {
        panic!("Unsupported model kind: {}", model.kind);
    }
}
