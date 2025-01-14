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

#[derive(Clone)]
pub enum OpenAIRequestFieldEnum {
    PresencePenalty,
    User,
}

#[derive(Builder, Clone)]
pub struct ExtendedOpenAIConfig {
    base: OpenAIConfig,

    #[builder(setter(into))]
    model_name: String,

    #[builder(setter(into))]
    supported_models: Option<Vec<String>>,

    #[builder(default)]
    fields_to_remove: Vec<OpenAIRequestFieldEnum>,
}

impl ExtendedOpenAIConfig {
    pub fn builder() -> ExtendedOpenAIConfigBuilder {
        ExtendedOpenAIConfigBuilder::default()
    }

    pub fn mistral_fields_to_remove() -> Vec<OpenAIRequestFieldEnum> {
        vec![
            OpenAIRequestFieldEnum::PresencePenalty,
            OpenAIRequestFieldEnum::User,
        ]
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

        for field in &self.fields_to_remove {
            match field {
                OpenAIRequestFieldEnum::PresencePenalty => {
                    request.presence_penalty = None;
                }
                OpenAIRequestFieldEnum::User => {
                    request.user = None;
                }
            }
        }

        request
    }
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
        self.chat().create(request).await
    }

    async fn chat_stream(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<ChatCompletionResponseStream, OpenAIError> {
        self.chat().create_stream(request).await
    }
}
