mod openai_chat;

use std::sync::Arc;

use openai_chat::OpenAIChatEngine;
use tabby_common::config::HttpModelConfig;
use tabby_inference::ChatCompletionStream;

pub fn create(model: &HttpModelConfig) -> Arc<dyn ChatCompletionStream> {
    if model.kind == "openai-chat" {
        let engine = OpenAIChatEngine::create(
            &model.api_endpoint,
            model.model_name.as_deref().unwrap_or_default(),
            model.api_key.clone(),
        );
        Arc::new(engine)
    } else {
        panic!("Only openai-chat are supported for http chat");
    }
}
