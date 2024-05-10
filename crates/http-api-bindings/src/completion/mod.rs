mod llama;
mod openai;

use std::sync::Arc;

use llama::LlamaCppEngine;
use openai::OpenAIEngine;

use tabby_inference::{CompletionStream};

use crate::{get_optional_param, get_param};

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
        let engine = LlamaCppEngine::create(&api_endpoint, api_key);
        (Arc::new(engine), prompt_template, chat_template)
    } else {
        panic!("Only openai are supported for http completion");
    }
}
