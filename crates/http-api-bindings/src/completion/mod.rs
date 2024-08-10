mod llama;
mod mistral;
mod openai;

use std::sync::Arc;

use llama::LlamaCppEngine;
use mistral::MistralFIMEngine;
use openai::OpenAICompletionEngine;
use tabby_common::config::HttpModelConfig;
use tabby_inference::CompletionStream;

pub async fn create(model: &HttpModelConfig) -> Arc<dyn CompletionStream> {
    match model.kind.as_str() {
        "llama.cpp/completion" => {
            let engine = LlamaCppEngine::create(
                model
                    .api_endpoint
                    .as_deref()
                    .expect("api_endpoint is required"),
                model.api_key.clone(),
            );
            Arc::new(engine)
        }
        "ollama/completion" => ollama_api_bindings::create_completion(model).await,
        "mistral/completion" => {
            let engine = MistralFIMEngine::create(
                model.api_endpoint.as_deref(),
                model.api_key.clone(),
                model.model_name.clone(),
            );
            Arc::new(engine)
        }
        "openai/completion" => {
            let engine = OpenAICompletionEngine::create(
                model.model_name.clone(),
                model
                    .api_endpoint
                    .as_deref()
                    .expect("api_endpoint is required"),
                model.api_key.clone(),
            );
            Arc::new(engine)
        }
        unsupported_kind => panic!(
            "Unsupported model kind for http completion: {}",
            unsupported_kind
        ),
    }
}

const FIM_TOKEN: &str = "<FIM>";
const FIM_TEMPLATE: &str = "{prefix}<FIM>{suffix}";

pub fn build_completion_prompt(model: &HttpModelConfig) -> (Option<String>, Option<String>) {
    if model.kind == "mistral/completion" || model.kind == "openai/completion" {
        (Some(FIM_TEMPLATE.to_owned()), None)
    } else {
        (model.prompt_template.clone(), model.chat_template.clone())
    }
}
