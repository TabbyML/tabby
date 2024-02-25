mod openai;
mod openai_chat;

use std::sync::Arc;

use openai::OpenAIEngine;
use openai_chat::OpenAIChatEngine;
use serde_json::Value;
use tabby_inference::{make_text_generation, TextGeneration};

pub fn create(model: &str) -> (Arc<dyn TextGeneration>, Option<String>, Option<String>) {
    let params = serde_json::from_str(model).expect("Failed to parse model string");
    let kind = get_param(&params, "kind");
    if kind == "openai" {
        let model_name = get_optional_param(&params, "model_name").unwrap_or_default();
        let api_endpoint = get_param(&params, "api_endpoint");
        let api_key = get_optional_param(&params, "api_key");
        let prompt_template = get_optional_param(&params, "prompt_template");
        let chat_template = get_optional_param(&params, "chat_template");
        let engine =
            make_text_generation(OpenAIEngine::create(&api_endpoint, &model_name, api_key));
        (Arc::new(engine), prompt_template, chat_template)
    } else if kind == "openai-chat" {
        let model_name = get_optional_param(&params, "model_name").unwrap_or_default();
        let api_endpoint = get_param(&params, "api_endpoint");
        let api_key = get_optional_param(&params, "api_key");

        let engine = make_text_generation(OpenAIChatEngine::create(
            &api_endpoint,
            &model_name,
            api_key,
        ));
        (
            Arc::new(engine),
            None,
            Some(OpenAIChatEngine::chat_template().to_owned()),
        )
    } else {
        panic!("Only openai are supported for http backend");
    }
}

fn get_param(params: &Value, key: &str) -> String {
    params
        .get(key)
        .unwrap_or_else(|| panic!("Missing {} field", key))
        .as_str()
        .expect("Type unmatched")
        .to_string()
}

fn get_optional_param(params: &Value, key: &str) -> Option<String> {
    params.get(key).map(|x| x.to_string())
}
