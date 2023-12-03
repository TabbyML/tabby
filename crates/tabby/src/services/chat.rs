mod chat_prompt;

use std::sync::Arc;

use async_stream::stream;
use chat_prompt::ChatPromptBuilder;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use tabby_inference::{TextGeneration, TextGenerationOptions, TextGenerationOptionsBuilder};
use tracing::debug;
use utoipa::ToSchema;

use super::model;
use crate::{fatal, Device};

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
    id: String,
    object: String,
    created: i64,
    model: String,
    system_fingerprint: String,
    choices: Vec<Choice>,
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct Choice {
    index: i32,
    delta: Message,
    finish_reason: Option<String>,
}

pub struct ChatService {
    engine: Arc<dyn TextGeneration>,
    prompt_builder: ChatPromptBuilder,
}

impl ChatService {
    fn new(engine: Arc<dyn TextGeneration>, chat_template: String) -> Self {
        Self {
            engine,
            prompt_builder: ChatPromptBuilder::new(chat_template),
        }
    }

    fn text_generation_options() -> TextGenerationOptions {
        TextGenerationOptionsBuilder::default()
            .max_input_length(2048)
            .max_decoding_length(1920)
            .sampling_temperature(0.1)
            .build()
            .unwrap()
    }

    pub async fn generate(
        &self,
        request: &ChatCompletionRequest,
    ) -> BoxStream<ChatCompletionChunk> {
        let prompt = self.prompt_builder.build(&request.messages);
        let options = Self::text_generation_options();
        debug!("PROMPT: {}", prompt);
        
        //TODO: pull these values from system values. 
        // ie. version number, current timestamp
        let id = "test-id";
        let model_name = "chat-model";
        let ts = 1694268190;
        let fingerprint = "0.7.0";

        let s = stream! {

            //TODO: need to return the first chunk with the role set.
            // following chunks should have no role in the delta
            for await chunk_content in self.engine.generate_stream(&prompt, options).await {
                yield ChatCompletionChunk {
                    id: id.to_string(),
                    object: "chat.completion.chunk".to_string(),
                    created: ts,
                    model: model_name.to_string(),
                    system_fingerprint: fingerprint.to_string(),
                    choices: vec![Choice {
                        index: 0,
                        delta: Message {
                            role: "".to_string(),
                            content: chunk_content,
                        },
                        finish_reason: None,
                    }],
                };
            
            }
        };

        //TOD: return the last chunk with no delta content and a finish_reason
        // s = stream! {
        //     yield ChatCompletionChunk {
        //     id: id.to_string(),
        //             object: "chat.completion.chunk".to_string(),
        //             created: ts,
        //             model: model_name.to_string(),
        //             system_fingerprint: fingerprint.to_string(),
        //             choices: vec![Choice {
        //                 index: 0,
        //                 delta: "",
        //                 finish_reason: "stop",
        //             }]         
        //     }
        // };

        Box::pin(s)
    }
}

pub async fn create_chat_service(model: &str, device: &Device, parallelism: u8) -> ChatService {
    let (engine, model::PromptInfo { chat_template, .. }) =
        model::load_text_generation(model, device, parallelism).await;

    let Some(chat_template) = chat_template else {
        fatal!("Chat model requires specifying prompt template");
    };

    ChatService::new(engine, chat_template)
}
