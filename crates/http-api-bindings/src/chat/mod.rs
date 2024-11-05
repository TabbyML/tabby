use std::sync::Arc;

use async_openai::config::OpenAIConfig;
use tabby_common::config::HttpModelConfig;
use tabby_inference::{ChatCompletionStream, ExtendedOpenAIConfig};

use crate::create_reqwest_client;

pub async fn create(model: &HttpModelConfig) -> Arc<dyn ChatCompletionStream> {
    let api_endpoint = model
        .api_endpoint
        .as_deref()
        .expect("api_endpoint is required");
    let config = OpenAIConfig::default()
        .with_api_base(api_endpoint)
        .with_api_key(model.api_key.clone().unwrap_or_default());

    let mut builder = ExtendedOpenAIConfig::builder();
    builder
        .base(config)
        .supported_models(model.supported_models.clone())
        .model_name(model.model_name.as_deref().expect("Model name is required"));

    if model.kind == "openai/chat" {
        // Do nothing
    } else if model.kind == "mistral/chat" {
        builder.fields_to_remove(ExtendedOpenAIConfig::mistral_fields_to_remove());
    } else {
        panic!("Unsupported model kind: {}", model.kind);
    }

    let config = builder.build().expect("Failed to build config");

    Arc::new(
        async_openai::Client::with_config(config)
            .with_http_client(create_reqwest_client(api_endpoint)),
    )
}
