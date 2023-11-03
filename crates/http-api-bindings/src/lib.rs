mod fastchat;
mod vertex_ai;

use fastchat::FastChatEngine;
use serde_json::Value;
use tabby_inference::TextGeneration;
use vertex_ai::VertexAIEngine;

pub fn create(model: &str) -> (Box<dyn TextGeneration>, String) {
    let params = serde_json::from_str(model).expect("Failed to parse model string");
    let kind = get_param(&params, "kind");
    let metafile = get_param(&params, "tabby_config");
    if kind == "vertex-ai" {
        let api_endpoint = get_param(&params, "api_endpoint");
        let authorization = get_param(&params, "authorization");
        let engine = Box::new(VertexAIEngine::create(
            api_endpoint.as_str(),
            authorization.as_str(),
        ));
        (engine, metafile)
    } else if kind == "fastchat" {
        let model_name = get_param(&params, "model_name");
        let api_endpoint = get_param(&params, "api_endpoint");
        let authorization = get_param(&params, "authorization");
        let engine = Box::new(FastChatEngine::create(
            api_endpoint.as_str(),
            model_name.as_str(),
            authorization.as_str(),
        ));
        (engine, metafile)
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
