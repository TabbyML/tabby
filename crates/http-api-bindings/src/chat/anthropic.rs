use std::sync::Arc;

use crate::create_reqwest_client;
use async_openai_alt::error::{ApiError, OpenAIError};
use async_openai_alt::types::{
    ChatChoiceStream, ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage,
    ChatCompletionRequestUserMessageContent, ChatCompletionResponseStream,
    ChatCompletionStreamResponseDelta, CreateChatCompletionRequest, CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
};
use async_trait::async_trait;
use futures::TryStreamExt;
use reqwest::Client;
use serde::de::Error;
use serde::{Deserialize, Serialize};
use tabby_inference::ChatCompletionStream;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use async_openai_alt::types::{
    ChatChoice as OpenAIChatChoice,
    ChatCompletionResponseMessage as OpenAIChatCompletionResponseMessage,
    CreateChatCompletionResponse as OpenAICreateChatCompletionResponse,
    FinishReason as OpenAIFinishReason, Role as OpenAIRole,
};

const HUMAN_PROMPT: &str = "<|Human|>";
const AI_PROMPT: &str = "<|Assistant|>";

#[derive(Debug, Serialize)]
struct AnthropicCreateChatCompletionRequest {
    prompt: String,
    model: String,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    stop_sequences: Option<Vec<String>>,
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct AnthropicCreateChatCompletionStreamResponse {
    completion: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", content = "data")]
enum AnthropicEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: Message },

    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: u32,
        content_block: ContentBlock,
    },

    #[serde(rename = "ping")]
    Ping,

    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        index: u32,
        delta: ContentBlockDelta,
    },

    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: u32 },

    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: MessageDelta,
        usage: Option<Usage>,
    },

    #[serde(rename = "message_stop")]
    MessageStop,
}

#[derive(Debug, Deserialize)]
struct Message {
    id: String,
    #[serde(rename = "type")]
    type_: String,
    role: String,
    model: String,
    content: Vec<ContentItem>,
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    type_: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct ContentBlockDelta {
    #[serde(rename = "type")]
    type_: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct MessageDelta {
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ContentItem {
    #[serde(rename = "type")]
    type_: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct Usage {
    input_tokens: u32,
    cache_creation_input_tokens: u32,
    cache_read_input_tokens: u32,
    output_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct AnthropicCreateChatCompletionResponse {
    id: String,
    #[serde(rename = "type")]
    type_: String,
    role: String,
    model: String,
    content: Vec<ContentItem>,
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct CompletionUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct ServiceTierResponse {
    pub name: String,
    pub resource_group: String,
    pub location: String,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct CreateChatCompletionStreamChoice {
    pub delta: Option<CreateChatCompletionDelta>,
    pub index: u32,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize, Default)]
pub struct CreateChatCompletionDelta {
    pub content: Option<String>,
}

#[derive(Clone)]
pub struct AnthropicChatCompletion {
    client: Arc<Client>,
    api_endpoint: String,
    api_key: String,
    default_model: String,
}

impl AnthropicChatCompletion {
    pub fn new(api_endpoint: &str, api_key: &str, default_model: &str) -> Self {
        Self {
            client: Arc::new(create_reqwest_client(api_endpoint)),
            api_endpoint: api_endpoint.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            default_model: default_model.to_string(),
        }
    }

    fn completion_url(&self) -> String {
        format!("{}/complete", self.api_endpoint)
    }

    fn transform_request(
        &self,
        request: &CreateChatCompletionRequest,
    ) -> Result<AnthropicCreateChatCompletionRequest, OpenAIError> {
        let prompt = request
            .messages
            .iter()
            .filter_map(|msg| match msg {
                ChatCompletionRequestMessage::User(user_msg) => match &user_msg.content {
                    ChatCompletionRequestUserMessageContent::Text(text) => Some(text.clone()),
                    _ => None,
                },
                ChatCompletionRequestMessage::Assistant(assistant_msg) => {
                    match &assistant_msg.content {
                        Some(ChatCompletionRequestAssistantMessageContent::Text(text)) => {
                            Some(text.clone())
                        }
                        _ => None,
                    }
                }
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        Ok(AnthropicCreateChatCompletionRequest {
            prompt: format!("{}{}\n{}", HUMAN_PROMPT, prompt, AI_PROMPT),
            model: request.model.clone(),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            stop_sequences: Some(vec![HUMAN_PROMPT.to_string()]),
            stream: false,
        })
    }

    fn transform_response(
        &self,
        anthropic_response: AnthropicCreateChatCompletionResponse,
    ) -> Result<OpenAICreateChatCompletionResponse, OpenAIError> {
        let completion_text = anthropic_response
            .content
            .iter()
            .filter_map(|item| {
                if item.type_ == "text" {
                    Some(item.text.clone())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        if completion_text.is_empty() {
            return Err(OpenAIError::ApiError(ApiError {
                message: "No completion found".to_string(),
                r#type: None,
                param: None,
                code: None,
            }));
        }

        Ok(OpenAICreateChatCompletionResponse {
            id: anthropic_response.id.clone(),
            choices: vec![OpenAIChatChoice {
                message: OpenAIChatCompletionResponseMessage {
                    role: OpenAIRole::Assistant,
                    content: Some(completion_text),
                    refusal: None,
                    tool_calls: None,
                    function_call: None,
                },
                finish_reason: match anthropic_response.stop_reason.as_deref() {
                    Some("end_turn") => Some(OpenAIFinishReason::Stop),
                    Some("length") => Some(OpenAIFinishReason::Length),
                    Some("content_filter") => Some(OpenAIFinishReason::ContentFilter),
                    _ => None,
                },
                index: 0,
                logprobs: None,
            }],
            created: 0,
            model: anthropic_response.model.clone(),
            service_tier: None,
            system_fingerprint: None,
            object: "chat.completion".to_string(),
            usage: None,
        })
    }

    fn transform_stream_chunk(
        &self,
        anthropic_chunk: AnthropicCreateChatCompletionStreamResponse,
    ) -> Option<CreateChatCompletionStreamChoice> {
        anthropic_chunk
            .completion
            .map(|part| CreateChatCompletionStreamChoice {
                delta: Some(CreateChatCompletionDelta {
                    content: Some(part),
                    ..Default::default()
                }),
                index: 0,
                finish_reason: None,
            })
    }

    fn transform_stream_chunk_static(
        anthropic_chunk: &AnthropicCreateChatCompletionStreamResponse,
    ) -> Option<CreateChatCompletionStreamChoice> {
        anthropic_chunk
            .completion
            .as_ref()
            .map(|part| CreateChatCompletionStreamChoice {
                delta: Some(CreateChatCompletionDelta {
                    content: Some(part.clone()),
                    ..Default::default()
                }),
                index: 0,
                finish_reason: None,
            })
    }
}

#[derive(Debug, Deserialize)]
struct WrappedError {
    pub error: ApiError,
}

#[async_trait]
impl ChatCompletionStream for AnthropicChatCompletion {
    async fn chat(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, OpenAIError> {
        let anthropic_request = self.transform_request(&request)?;

        let response = self
            .client
            .post(&self.completion_url())
            .header("Content-Type", "application/json")
            .header("X-API-Key", &self.api_key)
            .json(&anthropic_request)
            .send()
            .await
            .map_err(OpenAIError::Reqwest)?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            let api_error = serde_json::from_str::<WrappedError>(&error_text).map_err(|_| {
                OpenAIError::ApiError(ApiError {
                    message: error_text.clone(),
                    r#type: None,
                    param: None,
                    code: None,
                })
            })?;
            return Err(OpenAIError::ApiError(api_error.error));
        }

        let anthropic_response: AnthropicCreateChatCompletionResponse = response
            .json()
            .await
            .map_err(|e| OpenAIError::JSONDeserialize(serde_json::Error::custom(e)))?;

        self.transform_response(anthropic_response)
    }

    async fn chat_stream(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<ChatCompletionResponseStream, OpenAIError> {
        let anthropic_request = self.transform_request(&request)?;

        let anthropic_request_stream = AnthropicCreateChatCompletionRequest {
            prompt: format!(
                "{}{}\n{}",
                HUMAN_PROMPT, anthropic_request.prompt, AI_PROMPT
            ),
            model: anthropic_request.model,
            max_tokens: anthropic_request.max_tokens,
            temperature: anthropic_request.temperature,
            stop_sequences: anthropic_request.stop_sequences,
            stream: true,
        };

        let response = self
            .client
            .post(&self.completion_url())
            .header("Content-Type", "application/json")
            .header("x-api-key", &self.api_key)
            .json(&anthropic_request_stream)
            .send()
            .await
            .map_err(OpenAIError::Reqwest)?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            let api_error = serde_json::from_str::<WrappedError>(&error_text).map_err(|_| {
                OpenAIError::ApiError(ApiError {
                    message: error_text.clone(),
                    r#type: None,
                    param: None,
                    code: None,
                })
            })?;
            return Err(OpenAIError::ApiError(api_error.error));
        }

        let (tx, rx) = mpsc::channel(100);

        let mut stream = response
            .bytes_stream()
            .map_err(|e| OpenAIError::StreamError(e.to_string()));

        let tx_clone = tx.clone();
        tokio::spawn(async move {
            let mut response_id: Option<String> = None;
            let mut created_timestamp: Option<u32> = None;
            let mut model_name: Option<String> = None;

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        let chunk_str = match String::from_utf8(chunk.to_vec()) {
                            Ok(s) => s,
                            Err(_) => continue,
                        };

                        for line in chunk_str.lines() {
                            if line.starts_with("event: ") {
                            } else if line.starts_with("data: ") {
                                let data_str = line["data: ".len()..].trim();
                                if data_str.is_empty() {
                                    continue;
                                }

                                match serde_json::from_str::<AnthropicEvent>(data_str) {
                                    Ok(event) => {
                                        match event {
                                            AnthropicEvent::MessageStart { message } => {
                                                response_id = Some(message.id.clone());
                                                created_timestamp = Some(0);
                                                model_name = Some(message.model.clone());
                                            }
                                            AnthropicEvent::ContentBlockDelta { delta, .. } => {
                                                let text = delta.text.clone();
                                                let chat_choice = ChatChoiceStream {
                                                    delta: ChatCompletionStreamResponseDelta {
                                                        content: Some(text),
                                                        role: Some(async_openai_alt::types::Role::Assistant),
                                                        refusal: None,
                                                        tool_calls: None,
                                                        function_call: None,
                                                    },
                                                    index: 0,
                                                    finish_reason: None,
                                                    logprobs: None,
                                                };

                                                let response = CreateChatCompletionStreamResponse {
                                                    id: response_id.clone().unwrap_or_default(),
                                                    choices: vec![chat_choice],
                                                    created: created_timestamp.unwrap_or(0),
                                                    model: model_name.clone().unwrap_or_default(),
                                                    service_tier: None,
                                                    system_fingerprint: None,
                                                    object: "chat.completion.chunk".to_string(),
                                                    usage: None,
                                                };

                                                if tx_clone.send(Ok(response)).await.is_err() {
                                                    break;
                                                }
                                            }
                                            AnthropicEvent::MessageDelta { delta, usage } => {
                                                let chat_choice = ChatChoiceStream {
                                                    delta: ChatCompletionStreamResponseDelta {
                                                        content: None,
                                                        role: Some(async_openai_alt::types::Role::Assistant),
                                                        refusal: None,
                                                        tool_calls: None,
                                                        function_call: None,
                                                    },
                                                    index: 0,
                                                    finish_reason: Some(async_openai_alt::types::FinishReason::Stop),
                                                    logprobs: None,
                                                };

                                                let response = CreateChatCompletionStreamResponse {
                                                    id: response_id.clone().unwrap_or_default(),
                                                    choices: vec![chat_choice],
                                                    created: created_timestamp.unwrap_or(0),
                                                    model: model_name.clone().unwrap_or_default(),
                                                    service_tier: None,
                                                    system_fingerprint: None,
                                                    object: "chat.completion.chunk".to_string(),
                                                    usage: None,
                                                };

                                                if tx_clone.send(Ok(response)).await.is_err() {
                                                    break;
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
                                    Err(e) => {
                                        let _ = tx_clone
                                            .send(Err(OpenAIError::JSONDeserialize(e)))
                                            .await;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx_clone
                            .send(Err(OpenAIError::StreamError(e.to_string())))
                            .await;
                        break;
                    }
                }
            }
        });

        Ok(Box::pin(ReceiverStream::new(rx)))
    }
}
