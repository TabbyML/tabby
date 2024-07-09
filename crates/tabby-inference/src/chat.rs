use async_openai::{
    config::OpenAIConfig,
    error::OpenAIError,
    types::{
        ChatCompletionResponseStream, CreateChatCompletionRequest, CreateChatCompletionResponse,
    },
};
use async_trait::async_trait;
use derive_builder::Builder;

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
        request.model = self.model_name.clone();

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
        eprintln!("Creating chat stream: {:?}", request);
        self.chat().create_stream(request).await
    }
}
