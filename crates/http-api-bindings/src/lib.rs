mod fastchat;
mod vertex_ai;

use std::sync::Arc;

use fastchat::FastChatEngine;
use serde_json::Value;
use tabby_inference::TextGeneration;
use vertex_ai::VertexAIEngine;

pub fn create(model: &str) -> (Arc<dyn TextGeneration>, String) {
    let params = serde_json::from_str(model).expect("Failed to parse model string");
    let kind = get_param(&params, "kind");
    if kind == "vertex-ai" {
        let api_endpoint = get_param(&params, "api_endpoint");
        let authorization = get_param(&params, "authorization");
        let engine = VertexAIEngine::create(api_endpoint.as_str(), authorization.as_str());
        (Arc::new(engine), VertexAIEngine::prompt_template())
    } else if kind == "fastchat" {
        let model_name = get_param(&params, "model_name");
        let api_endpoint = get_param(&params, "api_endpoint");
        let authorization = get_param(&params, "authorization");
        let engine = FastChatEngine::create(
            api_endpoint.as_str(),
            model_name.as_str(),
            authorization.as_str(),
        );
        (Arc::new(engine), FastChatEngine::prompt_template())
    } else {
        panic!("Only vertex_ai and fastchat are supported for http backend");
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
