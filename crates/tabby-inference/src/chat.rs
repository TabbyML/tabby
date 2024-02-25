use anyhow::{bail, Result};
use async_openai::types::{
    ChatChoiceStream, ChatCompletionStreamResponseDelta, CreateChatCompletionRequest,
    CreateChatCompletionStreamResponse,
};
use async_stream::stream;
use async_trait::async_trait;
use futures::stream::BoxStream;
use uuid::Uuid;

use crate::{TextGenerationOptions, TextGenerationOptionsBuilder, TextGenerationStream};

#[async_trait]
pub trait ChatCompletionStreaming: Sync + Send {
    async fn chat_completion(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<BoxStream<CreateChatCompletionStreamResponse>>;
}

pub trait ChatPromptBuilder {
    fn build_chat_prompt(&self, request: CreateChatCompletionRequest) -> Result<String>;
}

#[async_trait]
impl<T: ChatPromptBuilder + TextGenerationStream> ChatCompletionStreaming for T {
    async fn chat_completion(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<BoxStream<CreateChatCompletionStreamResponse>> {
        let seed = request
            .seed
            .map(|x| x as u64)
            .unwrap_or_else(TextGenerationOptions::default_seed);

        let options = TextGenerationOptionsBuilder::default()
            .max_input_length(2048)
            .max_decoding_length(request.max_tokens.unwrap_or(1920).into())
            .seed(seed)
            .sampling_temperature(request.temperature.unwrap_or(0.1))
            .build()?;

        let prompt = self.build_chat_prompt(request)?;

        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Must be able to read system clock")
            .as_secs() as u32;
        let id = format!("chatcmpl-{}", Uuid::new_v4());
        let s = stream! {
            for await content in self.generate(&prompt, options).await {
                let delta = ChatCompletionStreamResponseDelta {
                    content:Some(content),
                    function_call: None,
                    tool_calls: None,
                    role: Some(async_openai::types::Role::Assistant)
                };
                let resp = CreateChatCompletionStreamResponse {
                    id: id.clone(),
                    choices: vec![ChatChoiceStream {
                        index: 0,
                        delta,
                        finish_reason: None,
                        logprobs: None,
                    }],
                    created,
                    model: "unused-model".into(),
                    system_fingerprint: None,
                    object: "chat.completion.chunk".into(),
                };

                yield resp;
            }
        };

        Ok(Box::pin(s))
    }
}
