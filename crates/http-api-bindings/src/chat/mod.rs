use std::sync::Arc;

use async_openai_alt::config::OpenAIConfig;
use tabby_common::config::HttpModelConfig;
use tabby_inference::{ChatCompletionStream, ExtendedOpenAIConfig};

use super::rate_limit;
use crate::{create_reqwest_client, AZURE_API_VERSION};

pub async fn create(model: &HttpModelConfig) -> Arc<dyn ChatCompletionStream> {
    let api_endpoint = model
        .api_endpoint
        .as_deref()
        .expect("api_endpoint is required");

    let engine: Box<dyn ChatCompletionStream> = match model.kind.as_str() {
        "azure/chat" => {
            let config = async_openai_alt::config::AzureConfig::new()
                .with_api_base(api_endpoint)
                .with_api_key(model.api_key.clone().unwrap_or_default())
                .with_api_version(AZURE_API_VERSION.to_string())
                .with_deployment_id(model.model_name.as_deref().expect("Model name is required"));
            Box::new(
                async_openai_alt::Client::with_config(config)
                    .with_http_client(create_reqwest_client(api_endpoint)),
            )
        }
        "openai/chat" | "mistral/chat" => {
            let config = OpenAIConfig::default()
                .with_api_base(api_endpoint)
                .with_api_key(model.api_key.clone().unwrap_or_default());

            let mut builder = ExtendedOpenAIConfig::builder();
            builder
                .base(config)
                .kind(model.kind.clone())
                .supported_models(model.supported_models.clone())
                .model_name(model.model_name.as_deref().expect("Model name is required"));

            Box::new(
                async_openai_alt::Client::with_config(
                    builder.build().expect("Failed to build config"),
                )
                .with_http_client(create_reqwest_client(api_endpoint)),
            )
        }
        _ => panic!("Unsupported model kind: {}", model.kind),
    };

    Arc::new(rate_limit::new_chat(
        engine,
        model.rate_limit.request_per_minute,
    ))
}
