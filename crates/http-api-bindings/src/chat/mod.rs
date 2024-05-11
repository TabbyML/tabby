mod openai_chat;

use std::sync::Arc;

use openai_chat::OpenAIChatEngine;
use tabby_inference::ChatCompletionStream;

use crate::{get_optional_param, get_param};

pub fn create(model: &str) -> Arc<dyn ChatCompletionStream> {
    let params = serde_json::from_str(model).expect("Failed to parse model string");
    let kind = get_param(&params, "kind");
    if kind == "openai-chat" {
        let model_name = get_optional_param(&params, "model_name").unwrap_or_default();
        let api_endpoint = get_param(&params, "api_endpoint");
        let api_key = get_optional_param(&params, "api_key");

        let engine = OpenAIChatEngine::create(&api_endpoint, &model_name, api_key);
        Arc::new(engine)
    } else {
        panic!("Only openai-chat are supported for http chat");
    }
}
