mod prompt;

use std::sync::Arc;

use async_stream::stream;
use futures::stream::BoxStream;
use prompt::ChatPromptBuilder;
use serde::{Deserialize, Serialize};
use tabby_common::languages::EMPTY_LANGUAGE;
use tabby_inference::{TextGeneration, TextGenerationOptions, TextGenerationOptionsBuilder};
use tracing::debug;
use utoipa::ToSchema;

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
#[schema(example=json!({
    "messages": [
        Message { role: "user".to_owned(), content: "What is tail recursion?".to_owned()},
        Message { role: "assistant".to_owned(), content: "It's a kind of optimization in compiler?".to_owned()},
        Message { role: "user".to_owned(), content: "Could you share more details?".to_owned()},
    ]
}))]
pub struct ChatCompletionRequest {
    messages: Vec<Message>,
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct Message {
    role: String,
    content: String,
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct ChatCompletionChunk {
    content: String,
}

pub struct ChatService {
    engine: Arc<Box<dyn TextGeneration>>,
    prompt_builder: ChatPromptBuilder,
}

impl ChatService {
    pub fn new(engine: Arc<Box<dyn TextGeneration>>, chat_template: String) -> Self {
        Self {
            engine,
            prompt_builder: ChatPromptBuilder::new(chat_template),
        }
    }

    fn parse_request(&self, request: &ChatCompletionRequest) -> (String, TextGenerationOptions) {
        let mut builder = TextGenerationOptionsBuilder::default();

        builder
            .max_input_length(2048)
            .max_decoding_length(1920)
            .language(&EMPTY_LANGUAGE)
            .sampling_temperature(0.1);

        (
            self.prompt_builder.build(&request.messages),
            builder.build().unwrap(),
        )
    }

    pub async fn generate(
        &self,
        request: &ChatCompletionRequest,
    ) -> BoxStream<ChatCompletionChunk> {
        let (prompt, options) = self.parse_request(request);
        debug!("PROMPT: {}", prompt);
        let s = stream! {
            for await content in self.engine.generate_stream(&prompt, options).await {
                yield ChatCompletionChunk { content }
            }
        };

        Box::pin(s)
    }
}
