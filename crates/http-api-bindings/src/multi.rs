use async_openai_alt::{
    error::OpenAIError,
    types::{
        ChatCompletionResponseStream, CreateChatCompletionRequest, CreateChatCompletionResponse,
    },
};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tabby_inference::ChatCompletionStream;

struct ChatStreamWrapper {
    model_name: String,
    chat_stream: Arc<dyn ChatCompletionStream>,
}

impl ChatStreamWrapper {
    fn new(model_name: String, chat_stream: Arc<dyn ChatCompletionStream>) -> Self {
        Self {
            model_name,
            chat_stream,
        }
    }

    fn process_request(
        &self,
        mut request: CreateChatCompletionRequest,
    ) -> CreateChatCompletionRequest {
        request.model = self.model_name.clone();
        request
    }
}

#[async_trait]
impl ChatCompletionStream for ChatStreamWrapper {
    async fn chat(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, OpenAIError> {
        let request = self.process_request(request);
        self.chat_stream.chat(request).await
    }

    async fn chat_stream(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<ChatCompletionResponseStream, OpenAIError> {
        let request = self.process_request(request);
        self.chat_stream.chat_stream(request).await
    }
}

pub struct MultiChatStream {
    chat_streams: HashMap<String, Box<dyn ChatCompletionStream>>,

    /// Provide a default value to handle the scenario when the request model is None,
    /// which is usually the model value from the first [add_chat_stream]
    default_model: Option<String>,
}

impl MultiChatStream {
    pub fn new() -> MultiChatStream {
        Self {
            chat_streams: HashMap::new(),
            default_model: None,
        }
    }

    pub fn add_chat_stream(
        &mut self,
        model_title: impl Into<String>,
        model: impl Into<String>,
        completion: Arc<dyn ChatCompletionStream>,
    ) {
        let model_title = model_title.into();
        if self.default_model.is_none() {
            self.default_model = Some(model_title.to_owned());
        }
        self.chat_streams.insert(
            model_title,
            Box::new(ChatStreamWrapper::new(model.into(), completion)),
        );
    }

    fn get_chat_stream(&self, model: &str) -> Result<&Box<dyn ChatCompletionStream>, OpenAIError> {
        let model = if model.is_empty() {
            self.default_model
                .as_ref()
                .ok_or_else(|| OpenAIError::InvalidArgument("No available model".to_owned()))?
        } else {
            model
        };
        self.chat_streams
            .get(model)
            .ok_or_else(|| OpenAIError::InvalidArgument(format!("Model {} does not exist", model)))
    }
}

#[async_trait]
impl ChatCompletionStream for MultiChatStream {
    async fn chat(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, OpenAIError> {
        let chat_stream = self.get_chat_stream(&request.model)?;
        chat_stream.chat(request).await
    }

    async fn chat_stream(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<ChatCompletionResponseStream, OpenAIError> {
        let chat_stream = self.get_chat_stream(&request.model)?;
        chat_stream.chat_stream(request).await
    }
}
