use async_openai::{
    error::OpenAIError,
    types::{
        ChatChoice, ChatChoiceStream, ChatCompletionResponseMessage, ChatCompletionResponseStream,
        ChatCompletionStreamResponseDelta, CompletionUsage, CreateChatCompletionRequest,
        CreateChatCompletionResponse, CreateChatCompletionStreamResponse, FinishReason, Role,
    },
};
use axum::async_trait;
use tabby_common::api::{
    code::{CodeSearch, CodeSearchError, CodeSearchParams, CodeSearchQuery, CodeSearchResponse},
    doc::{DocSearch, DocSearchDocument, DocSearchError, DocSearchHit, DocSearchResponse},
};
use tabby_inference::ChatCompletionStream;
use tabby_schema::{
    context::{ContextInfo, ContextService},
    policy::AccessPolicy,
    Result,
};

pub struct FakeChatCompletionStream;
#[async_trait]
impl ChatCompletionStream for FakeChatCompletionStream {
    async fn chat(
        &self,
        _request: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, OpenAIError> {
        Ok(CreateChatCompletionResponse {
            id: "test-response".to_owned(),
            created: 0,
            model: "ChatTabby".to_owned(),
            object: "chat.completion".to_owned(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatCompletionResponseMessage {
                    role: Role::Assistant,
                    content: Some(
                        "1. What is the main functionality of the provided code?\n\
                             2. How does the code snippet implement a web server?\n\
                             3. Can you explain how the Flask app works in this context?"
                            .to_string(),
                    ),
                    tool_calls: None,
                    function_call: None,
                },
                finish_reason: Some(FinishReason::Stop),
                logprobs: None,
            }],
            system_fingerprint: Some("seed".to_owned()),
            usage: Some(CompletionUsage {
                prompt_tokens: 1,
                completion_tokens: 2,
                total_tokens: 3,
            }),
        })
    }

    async fn chat_stream(
        &self,
        _request: CreateChatCompletionRequest,
    ) -> Result<ChatCompletionResponseStream, OpenAIError> {
        let stream = futures::stream::iter(vec![
            Ok(CreateChatCompletionStreamResponse {
                id: "test-stream-response".to_owned(),
                created: 0,
                model: "ChatTabby".to_owned(),
                object: "chat.completion.chunk".to_owned(),
                choices: vec![ChatChoiceStream {
                    index: 0,
                    delta: ChatCompletionStreamResponseDelta {
                        role: Some(Role::Assistant),
                        content: Some("This is the first part of the response. ".to_string()),
                        function_call: None,
                        tool_calls: None,
                    },
                    finish_reason: None,
                    logprobs: None,
                }],
                system_fingerprint: Some("seed".to_owned()),
            }),
            Ok(CreateChatCompletionStreamResponse {
                id: "test-stream-response".to_owned(),
                created: 0,
                model: "ChatTabby".to_owned(),
                object: "chat.completion.chunk".to_owned(),
                choices: vec![ChatChoiceStream {
                    index: 0,
                    delta: ChatCompletionStreamResponseDelta {
                        role: None,
                        content: Some("This is the second part of the response.".to_string()),
                        function_call: None,
                        tool_calls: None,
                    },
                    finish_reason: Some(FinishReason::Stop),
                    logprobs: None,
                }],
                system_fingerprint: Some("seed".to_owned()),
            }),
        ]);

        Ok(Box::pin(stream) as ChatCompletionResponseStream)
    }
}
pub struct FakeCodeSearch;

#[async_trait]
impl CodeSearch for FakeCodeSearch {
    async fn search_in_language(
        &self,
        _query: CodeSearchQuery,
        _params: CodeSearchParams,
    ) -> Result<CodeSearchResponse, CodeSearchError> {
        Ok(CodeSearchResponse { hits: vec![] })
    }
}

pub struct FakeCodeSearchFailNotReady;
#[async_trait]
impl CodeSearch for FakeCodeSearchFailNotReady {
    async fn search_in_language(
        &self,
        _query: CodeSearchQuery,
        _params: CodeSearchParams,
    ) -> Result<CodeSearchResponse, CodeSearchError> {
        Err(CodeSearchError::NotReady)
    }
}

pub struct FakeCodeSearchFail;
#[async_trait]
impl CodeSearch for FakeCodeSearchFail {
    async fn search_in_language(
        &self,
        _query: CodeSearchQuery,
        _params: CodeSearchParams,
    ) -> Result<CodeSearchResponse, CodeSearchError> {
        Err(CodeSearchError::Other(anyhow::anyhow!("error")))
    }
}

pub struct FakeDocSearch;
#[async_trait]
impl DocSearch for FakeDocSearch {
    async fn search(
        &self,
        _source_ids: &[String],
        _q: &str,
        _limit: usize,
    ) -> Result<DocSearchResponse, DocSearchError> {
        let hits = vec![
            DocSearchHit {
                score: 1.0,
                doc: DocSearchDocument {
                    title: "Document 1".to_string(),
                    link: "https://example.com/doc1".to_string(),
                    snippet: "Snippet for Document 1".to_string(),
                },
            },
            DocSearchHit {
                score: 0.9,
                doc: DocSearchDocument {
                    title: "Document 2".to_string(),
                    link: "https://example.com/doc2".to_string(),
                    snippet: "Snippet for Document 2".to_string(),
                },
            },
            DocSearchHit {
                score: 0.8,
                doc: DocSearchDocument {
                    title: "Document 3".to_string(),
                    link: "https://example.com/doc3".to_string(),
                    snippet: "Snippet for Document 3".to_string(),
                },
            },
            DocSearchHit {
                score: 0.7,
                doc: DocSearchDocument {
                    title: "Document 4".to_string(),
                    link: "https://example.com/doc4".to_string(),
                    snippet: "Snippet for Document 4".to_string(),
                },
            },
            DocSearchHit {
                score: 0.6,
                doc: DocSearchDocument {
                    title: "Document 5".to_string(),
                    link: "https://example.com/doc5".to_string(),
                    snippet: "Snippet for Document 5".to_string(),
                },
            },
        ];
        Ok(DocSearchResponse { hits })
    }
}

pub struct FakeContextService;
#[async_trait]
impl ContextService for FakeContextService {
    async fn read(&self, _policy: Option<&AccessPolicy>) -> Result<ContextInfo> {
        Ok(ContextInfo { sources: vec![] })
    }
}
