use async_openai::{
    config::OpenAIConfig,
    error::OpenAIError,
    types::{
        ChatCompletionResponseStream, CreateChatCompletionRequest, CreateChatCompletionResponse,
    },
};
use async_trait::async_trait;

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

#[derive(Clone)]
pub struct ExtendedOpenAIConfig {
    base: OpenAIConfig,
    default_model: String,
}

impl ExtendedOpenAIConfig {
    pub fn new(base: OpenAIConfig, default_model: String) -> Self {
        Self {
            base,
            default_model,
        }
    }
}

impl async_openai::config::Config for ExtendedOpenAIConfig {
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
impl ChatCompletionStream for async_openai::Client<ExtendedOpenAIConfig> {
    async fn chat(
        &self,
        mut request: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, OpenAIError> {
        request.model = self.config().default_model.clone();
        self.chat().create(request).await
    }

    async fn chat_stream(
        &self,
        mut request: CreateChatCompletionRequest,
    ) -> Result<ChatCompletionResponseStream, OpenAIError> {
        request.model = self.config().default_model.clone();
        self.chat().create_stream(request).await
    }
}
