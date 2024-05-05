mod llama;
mod openai;
mod openai_chat;

use std::sync::Arc;

use openai::OpenAIEngine;
use openai_chat::OpenAIChatEngine;
use serde_json::Value;
use tabby_inference::{ChatCompletionStream, CompletionStream};

pub fn create(model: &str) -> (Arc<dyn CompletionStream>, Option<String>, Option<String>) {
    let params = serde_json::from_str(model).expect("Failed to parse model string");
    let kind = get_param(&params, "kind");
    if kind == "openai" {
        let model_name = get_optional_param(&params, "model_name").unwrap_or_default();
        let api_endpoint = get_param(&params, "api_endpoint");
        let api_key = get_optional_param(&params, "api_key");
        let prompt_template = get_optional_param(&params, "prompt_template");
        let chat_template = get_optional_param(&params, "chat_template");
        let engine = OpenAIEngine::create(&api_endpoint, &model_name, api_key);
        (Arc::new(engine), prompt_template, chat_template)
    } else if kind == "llama" {
        let api_endpoint = get_param(&params, "api_endpoint");
        let api_key = get_optional_param(&params, "api_key");
        let prompt_template = get_optional_param(&params, "prompt_template");
        let chat_template = get_optional_param(&params, "chat_template");
        let engine = llama::LlamaCppEngine::create(&api_endpoint, api_key);
        (Arc::new(engine), prompt_template, chat_template)
    } else {
        panic!("Only openai are supported for http completion");
    }
}

pub fn create_chat(model: &str) -> Arc<dyn ChatCompletionStream> {
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

fn get_param(params: &Value, key: &str) -> String {
    params
        .get(key)
        .unwrap_or_else(|| panic!("Missing {} field", key))
        .as_str()
        .expect("Type unmatched")
        .to_owned()
}

fn get_optional_param(params: &Value, key: &str) -> Option<String> {
    params
        .get(key)
        .map(|x| x.as_str().expect("Type unmatched").to_owned())
}
