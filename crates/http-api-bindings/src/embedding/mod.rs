mod llama;

use std::sync::Arc;

use llama::LlamaCppEngine;
use tabby_inference::Embedding;

use crate::{get_optional_param, get_param};

pub fn create(model: &str) -> Arc<dyn Embedding> {
    let params = serde_json::from_str(model).expect("Failed to parse model string");
    let kind = get_param(&params, "kind");
    if kind == "llama" {
        let api_endpoint = get_param(&params, "api_endpoint");
        let api_key = get_optional_param(&params, "api_key");
        let engine = LlamaCppEngine::create(&api_endpoint, api_key);
        Arc::new(engine)
    } else {
        panic!("Only llama are supported for http embedding");
    }
}
