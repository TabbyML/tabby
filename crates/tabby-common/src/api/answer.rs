use async_openai::types::{ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use super::{
    code::{CodeSearchHit, CodeSearchQuery},
    doc::DocSearchHit,
};

#[derive(Deserialize, ToSchema)]
#[schema(example=json!({
    "messages": [
        ChatCompletionRequestUserMessageArgs::default().content("What is tail recursion?".to_owned()).build().unwrap(),
    ],
}))]
pub struct AnswerRequest {
    #[serde(default)]
    pub user: Option<String>,

    pub messages: Vec<ChatCompletionRequestMessage>,

    /// Client-provided code snippet to run relevant code search and enrich the LLM request.
    #[serde(default)]
    pub code_query: Option<CodeSearchQuery>,

    /// Client-provided context to enrich the LLM request.
    #[serde(default)]
    pub code_snippets: Vec<AnswerCodeSnippet>,

    #[serde(default)]
    pub doc_query: bool,

    #[serde(default)]
    pub generate_relevant_questions: bool,

    #[serde(default)]
    pub collect_relevant_code_using_user_message: bool,
}

#[derive(Deserialize, ToSchema)]
pub struct AnswerCodeSnippet {
    pub language: Option<String>,
    pub filepath: Option<String>,
    pub content: String,
}

#[derive(Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum AnswerResponseChunk {
    RelevantCode(Vec<CodeSearchHit>),
    RelevantDocuments(Vec<DocSearchHit>),
    RelevantQuestions(Vec<String>),
    AnswerDelta(String),
}
