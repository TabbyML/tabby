use std::sync::Arc;

use async_openai_alt::config::OpenAIConfig;
use tabby_common::config::HttpModelConfig;
use tabby_inference::{ChatCompletionStream, ExtendedOpenAIConfig};

use super::multi::MultiChatStream;
use super::rate_limit;
use crate::{create_reqwest_client, AZURE_API_VERSION};

pub async fn create(model: &HttpModelConfig) -> Arc<dyn ChatCompletionStream> {
    let mut multi_chat_stream = MultiChatStream::new();
    add_engine(&mut multi_chat_stream, model);
    Arc::new(multi_chat_stream)
}

pub async fn create_multiple(models: &[HttpModelConfig]) -> Arc<dyn ChatCompletionStream> {
    let mut multi_chat_stream = MultiChatStream::new();
    models.iter().for_each(|model| {
        add_engine(&mut multi_chat_stream, model);
    });
    Arc::new(multi_chat_stream)
}

fn add_engine(multi_chat_stream: &mut MultiChatStream, model: &HttpModelConfig) {
    let (model_title, model_name) = model.model_title_and_name();
    let engine = Arc::new(rate_limit::new_chat(
        create_engine(model),
        model.rate_limit.request_per_minute,
    ));

    // Handle model_name first just to set default_model to it
    multi_chat_stream.add_chat_stream(model_title, model_name, engine.clone());

    if let (Some(supported_models), Some(model_name)) = (&model.supported_models, &model.model_name)
    {
        for m in supported_models.iter().filter(|m| model_name != *m) {
            multi_chat_stream.add_chat_stream(m, m, engine.clone());
        }
    }
}

fn create_engine(model: &HttpModelConfig) -> Box<dyn ChatCompletionStream> {
    let api_endpoint = model
        .api_endpoint
        .as_deref()
        .expect("api_endpoint is required");

    match model.kind.as_str() {
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
    }
}
