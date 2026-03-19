use async_openai_alt::{
    config::OpenAIConfig,
    error::OpenAIError,
    types::{
        ChatCompletionResponseStream, CreateChatCompletionRequest, CreateChatCompletionResponse,
    },
};
use async_trait::async_trait;
use derive_builder::Builder;
use tracing::warn;

#[async_trait]
pub trait ChatCompletionStream: Sync + Send {
    async fn chat(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, OpenAIError>;

    async fn chat_stream(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<ChatCompletionResponseStream, OpenAIError>;
}

#[derive(Builder, Clone)]
pub struct ExtendedOpenAIConfig {
    #[builder(default)]
    kind: String,

    base: OpenAIConfig,

    #[builder(setter(into))]
    model_name: String,

    #[builder(setter(into))]
    supported_models: Option<Vec<String>>,
}

impl ExtendedOpenAIConfig {
    pub fn builder() -> ExtendedOpenAIConfigBuilder {
        ExtendedOpenAIConfigBuilder::default()
    }

    fn process_request(
        &self,
        mut request: CreateChatCompletionRequest,
    ) -> CreateChatCompletionRequest {
        if request.model.is_empty() {
            request.model = self.model_name.clone();
        } else if let Some(supported_models) = &self.supported_models {
            if !supported_models.contains(&request.model) {
                warn!(
                    "Warning: {} model is not supported, falling back to {}",
                    request.model, self.model_name
                );
                request.model = self.model_name.clone();
            }
        }

        match self.kind.as_str() {
            "mistral/chat" => {
                request.presence_penalty = None;
                request.user = None;
                request.stream_options = None;
            }
            "openai/chat" => {
                request = process_request_openai(request);
            }
            "minimax/chat" => {
                // MiniMax temperature must be in (0.0, 1.0]; default to 1.0 if unset or zero
                request.temperature = Some(
                    request
                        .temperature
                        .map(|t| if t <= 0.0 { 1.0 } else { t.min(1.0) })
                        .unwrap_or(1.0),
                );
            }
            _ => {}
        }

        request
    }
}

fn process_request_openai(request: CreateChatCompletionRequest) -> CreateChatCompletionRequest {
    let mut request = request;

    // Check for specific O-series model prefixes
    if request.model.starts_with("o1") || request.model.starts_with("o3-mini") {
        request.presence_penalty = None;
        request.frequency_penalty = None;
    }

    request
}

impl async_openai_alt::config::Config for ExtendedOpenAIConfig {
    fn headers(&self) -> reqwest::header::HeaderMap {
        self.base.headers()
    }

    fn url(&self, path: &str) -> String {
        self.base.url(path)
    }

    fn query(&self) -> Vec<(&str, &str)> {
        self.base.query()
    }

    fn api_base(&self) -> &str {
        self.base.api_base()
    }

    fn api_key(&self) -> &secrecy::Secret<String> {
        self.base.api_key()
    }
}

#[async_trait]
impl ChatCompletionStream for async_openai_alt::Client<ExtendedOpenAIConfig> {
    async fn chat(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, OpenAIError> {
        let request = self.config().process_request(request);
        self.chat().create(request).await
    }

    async fn chat_stream(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<ChatCompletionResponseStream, OpenAIError> {
        let request = self.config().process_request(request);
        self.chat().create_stream(request).await
    }
}

#[async_trait]
impl ChatCompletionStream for async_openai_alt::Client<async_openai_alt::config::AzureConfig> {
    async fn chat(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, OpenAIError> {
        let request = process_request_openai(request);
        self.chat().create(request).await
    }

    async fn chat_stream(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<ChatCompletionResponseStream, OpenAIError> {
        let request = process_request_openai(request);
        self.chat().create_stream(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(kind: &str) -> ExtendedOpenAIConfig {
        ExtendedOpenAIConfig::builder()
            .base(OpenAIConfig::default())
            .kind(kind.to_string())
            .model_name("test-model")
            .supported_models(None)
            .build()
            .unwrap()
    }

    fn make_request(model: &str, temperature: Option<f32>) -> CreateChatCompletionRequest {
        let mut req = CreateChatCompletionRequest::default();
        req.model = model.to_string();
        req.temperature = temperature;
        req
    }

    #[test]
    fn test_minimax_temperature_clamped_to_default_when_zero() {
        let config = make_config("minimax/chat");
        let req = make_request("MiniMax-M2.7", Some(0.0));
        let processed = config.process_request(req);
        assert_eq!(processed.temperature, Some(1.0));
    }

    #[test]
    fn test_minimax_temperature_clamped_to_default_when_negative() {
        let config = make_config("minimax/chat");
        let req = make_request("MiniMax-M2.7", Some(-0.5));
        let processed = config.process_request(req);
        assert_eq!(processed.temperature, Some(1.0));
    }

    #[test]
    fn test_minimax_temperature_clamped_to_max_when_above_one() {
        let config = make_config("minimax/chat");
        let req = make_request("MiniMax-M2.7", Some(1.5));
        let processed = config.process_request(req);
        assert_eq!(processed.temperature, Some(1.0));
    }

    #[test]
    fn test_minimax_temperature_preserved_when_valid() {
        let config = make_config("minimax/chat");
        let req = make_request("MiniMax-M2.7", Some(0.7));
        let processed = config.process_request(req);
        assert_eq!(processed.temperature, Some(0.7));
    }

    #[test]
    fn test_minimax_temperature_defaults_when_none() {
        let config = make_config("minimax/chat");
        let req = make_request("MiniMax-M2.7", None);
        let processed = config.process_request(req);
        assert_eq!(processed.temperature, Some(1.0));
    }

    #[test]
    fn test_minimax_model_fallback_when_empty() {
        let config = make_config("minimax/chat");
        let req = make_request("", Some(0.5));
        let processed = config.process_request(req);
        assert_eq!(processed.model, "test-model");
    }
}
