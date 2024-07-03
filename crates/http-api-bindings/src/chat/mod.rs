use std::sync::Arc;

use async_openai::config::OpenAIConfig;
use tabby_common::config::HttpModelConfig;
use tabby_inference::{ChatCompletionStream, ExtendedOpenAIConfig};

pub async fn create(model: &HttpModelConfig) -> Arc<dyn ChatCompletionStream> {
    let config = OpenAIConfig::default()
        .with_api_base(model.api_endpoint.clone())
        .with_api_key(model.api_key.clone().unwrap_or_default());

    let config = ExtendedOpenAIConfig::new(
        config,
        model
            .model_name
            .as_deref()
            .expect("Model name is required")
            .to_owned(),
    );

    Arc::new(async_openai::Client::with_config(config))
}
