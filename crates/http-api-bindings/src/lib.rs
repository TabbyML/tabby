mod openai;
mod vertex_ai;

use std::sync::Arc;

use openai::OpenAIEngine;
use serde_json::Value;
use tabby_inference::{make_text_generation, TextGeneration};
use vertex_ai::VertexAIEngine;

pub fn create(model: &str) -> (Arc<dyn TextGeneration>, String) {
    let params = serde_json::from_str(model).expect("Failed to parse model string");
    let kind = get_param(&params, "kind");
    if kind == "vertex-ai" {
        let api_endpoint = get_param(&params, "api_endpoint");
        let authorization = get_param(&params, "authorization");
        let engine = make_text_generation(VertexAIEngine::create(
            api_endpoint.as_str(),
            authorization.as_str(),
        ));
        (Arc::new(engine), VertexAIEngine::prompt_template())
    } else if kind == "openai" {
        let model_name = get_param(&params, "model_name");
        let api_endpoint = get_param(&params, "api_endpoint");
        let authorization = get_optional_param(&params, "authorization");
        let prompt_template = get_param(&params, "prompt_template");
        let engine = make_text_generation(OpenAIEngine::create(
            api_endpoint.as_str(),
            model_name.as_str(),
            authorization,
        ));
        (Arc::new(engine), prompt_template)
    } else {
        panic!("Only vertex_ai and openai are supported for http backend");
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
