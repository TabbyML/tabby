mod llama;
mod mistral;

use std::sync::Arc;

use llama::LlamaCppEngine;
use mistral::MistralFIMEngine;
use tabby_common::config::HttpModelConfig;
use tabby_inference::CompletionStream;

pub async fn create(model: &HttpModelConfig) -> Arc<dyn CompletionStream> {
    match model.kind.as_str() {
        "llama.cpp/completion" => {
            let engine = LlamaCppEngine::create(&model.api_endpoint, model.api_key.clone());
            Arc::new(engine)
        }
        "ollama/completion" => ollama_api_bindings::create_completion(model).await,

        "mistral/completion" => {
            let engine = MistralFIMEngine::create(
                &model.api_endpoint,
                model.api_key.clone(),
                model.model_name.clone(),
            );
            Arc::new(engine)
        }

        unsupported_kind => panic!(
            "Unsupported model kind for http completion: {}",
            unsupported_kind
        ),
    }
}

pub fn build_completion_prompt(model: &HttpModelConfig) -> (Option<String>, Option<String>) {
    if model.kind == "mistral/completion" {
        (Some("{prefix}<FIM>{suffix}".to_owned()), None)
    } else {
        (model.prompt_template.clone(), model.chat_template.clone())
    }
}
