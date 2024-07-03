use async_openai::{
    config::OpenAIConfig,
    error::OpenAIError,
    types::{ChatCompletionResponseStream, CreateChatCompletionRequest},
};
use async_trait::async_trait;

#[async_trait]
pub trait ChatCompletionStream: Sync + Send {
    async fn chat_stream(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<ChatCompletionResponseStream, OpenAIError>;
}

pub struct ExtendedOpenAIConfig {
    base: OpenAIConfig,
}

#[async_trait]
impl ChatCompletionStream for async_openai::Client<OpenAIConfig> {
    async fn chat_stream(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<ChatCompletionResponseStream, OpenAIError> {
        self.chat().create_stream(request).await
    }
}
