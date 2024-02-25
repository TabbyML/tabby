use serde::{Deserialize, Serialize};
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
    pub messages: Vec<Message>,
    pub temperature: Option<f32>,
    pub seed: Option<u64>,
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct Message {
    role: String,
    content: String,
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct ChatCompletionChunk {
    id: String,
    created: u64,
    system_fingerprint: String,
    object: &'static str,
    model: &'static str,
    choices: [ChatCompletionChoice; 1],
}

#[derive(Serialize, Deserialize, Clone, Debug, ToSchema)]
pub struct ChatCompletionChoice {
    index: usize,
    #[serde(skip_serializing_if="Option::is_none")]
    logprobs: Option<String>,
    #[serde(skip_serializing_if="Option::is_none")]
    finish_reason: Option<String>,
    delta: ChatCompletionDelta,
}

#[derive(Serialize, Deserialize, Clone, Debug, ToSchema)]
pub struct ChatCompletionDelta {
    #[serde(skip_serializing_if="Option::is_none")]
    content: Option<String>,
}

impl ChatCompletionChunk {
    pub fn new(content: Option<String>, id: String, created: u64, last_chunk: bool) -> Self {
        ChatCompletionChunk {
            id,
            created,
            object: "chat.completion.chunk",
            model: "unused-model",
            system_fingerprint: "unused-system-fingerprint".into(),
            choices: [ChatCompletionChoice {
                index: 0,
                delta: ChatCompletionDelta { content },
                logprobs: None,
                finish_reason: last_chunk.then(|| "stop".into()),
            }],
        }
    }
}