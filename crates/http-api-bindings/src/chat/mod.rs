mod openai_chat;

use std::sync::Arc;

use openai_chat::OpenAIChatEngine;
use tabby_common::config::HttpModelConfig;
use tabby_inference::ChatCompletionStream;

pub async fn create(model: &HttpModelConfig) -> Arc<dyn ChatCompletionStream> {
    match model.kind.as_str() {
        "openai/chat" => Arc::new(OpenAIChatEngine::create(
            &model.api_endpoint,
            model.model_name.as_deref().unwrap_or_default(),
            model.api_key.clone(),
        )),
        "ollama/chat" => ollama_api_bindings::create_chat(model).await,

        unsupported_kind => panic!("Unsupported kind for http chat: {}", unsupported_kind),
    }
}
