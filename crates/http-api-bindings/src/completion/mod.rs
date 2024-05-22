mod llama;

use std::sync::Arc;

use llama::LlamaCppEngine;
use tabby_common::config::HttpModelConfig;
use tabby_inference::CompletionStream;

pub async fn create(model: &HttpModelConfig) -> Arc<dyn CompletionStream> {
    match model.kind.as_str() {
        "llama.cpp/completion" => {
            let engine = LlamaCppEngine::create(&model.api_endpoint, model.api_key.clone());
            Arc::new(engine)
        }
        "ollama" => ollama_api_bindings::create_completion(model).await,

        unsupported_kind => panic!(
            "Unsupported model kind for http completion: {}",
            unsupported_kind
        ),
    }
}
