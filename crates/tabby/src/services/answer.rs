use std::sync::Arc;

use anyhow::{Context, Result};
use async_stream::stream;
use axum::{
    extract::{Query, State},
    Json,
};
use futures::{stream::BoxStream, AsyncReadExt, FutureExt};
use hyper::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tabby_common::api::{
    chat,
    code::{CodeSearch, CodeSearchDocument},
    doc::{DocSearch, DocSearchDocument},
};
use tabby_inference::ChatCompletionStream;
use tracing::{debug, instrument, warn};
use utoipa::{IntoParams, ToSchema};

use crate::services::chat::{ChatCompletionRequestBuilder, ChatService};

use super::chat::ChatCompletionRequest;

#[derive(Deserialize, ToSchema)]
#[schema(example=json!({
    "messages": [
        chat::Message { role: "user".to_owned(), content: "What is tail recursion?".to_owned()},
    ]
}))]
pub struct AnswerRequest {
    messages: Vec<chat::Message>,
}

#[derive(Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum AnswerResponseChunk {
    RelevantDoc(DocSearchDocument),
    RelevantQuestion(String),
    AnswerChunk(String),
}
pub struct AnswerService {
    chat: Arc<ChatService>,
    doc: Arc<dyn DocSearch>,
}

impl AnswerService {
    fn new(chat: Arc<ChatService>, doc: Arc<dyn DocSearch>) -> Self {
        Self { chat, doc }
    }

    pub async fn answer<'a>(
        self: Arc<Self>,
        mut req: AnswerRequest,
    ) -> BoxStream<'a, AnswerResponseChunk> {
        let s = stream! {
            // 1. Collect sources given query, for now we only use the last message
            let query: &mut chat::Message = match req.messages.last_mut() {
                Some(query) => query,
                None => {
                    warn!("No query found in the request");
                    return;
                }
            };

            // 2. Generate relevant docs from the query
            // For now we only collect from DocSearch.
            let docs = match self.doc.search(&query.content, 20, 0).await {
                Ok(docs) => docs,
                Err(err) => {
                    warn!("Failed to search docs: {:?}", err);
                    return;
                }
            };

            for hit in &docs.hits {
                yield AnswerResponseChunk::RelevantDoc(hit.doc.clone());
            }

            // 3. Generate relevant answers from the query
            let snippets = docs.hits.iter().map(|hit| hit.doc.snippet.as_str()).collect::<Vec<_>>();
            for await question in self.generate_relevant_questions(&snippets, &query.content).await {
                yield AnswerResponseChunk::RelevantQuestion(question);
            }

            // 4. Generate override prompt from the query
            (*query).content = self.generate_prompt(&snippets, &query.content).await;

            // 5. Generate answer from the query
            let s = self.chat.clone().generate(ChatCompletionRequestBuilder::default()
                .messages(req.messages.clone())
                .build()
                .expect("Failed to create ChatCompletionRequest"))
                .await;

            for await chunk in s {
                yield AnswerResponseChunk::AnswerChunk(chunk.choices[0].delta.content.clone());
            }
        };

        Box::pin(s)
    }

    async fn generate_relevant_questions(
        &self,
        snippets: &[&str],
        question: &str,
    ) -> BoxStream<String> {
        let context = snippets.join("\n");
        let prompt = format!(
            r#"
You are a helpful assistant that helps the user to ask related questions, based on user's original question and the related contexts. Please identify worthwhile topics that can be follow-ups, and write questions no longer than 20 words each. Please make sure that specifics, like events, names, locations, are included in follow up questions so they can be asked standalone. For example, if the original question asks about "the Manhattan project", in the follow up question, do not just say "the project", but use the full name "the Manhattan project". Your related questions must be in the same language as the original question.

Here are the contexts of the question:

{context}

Remember, based on the original question and related contexts, suggest three such further questions. Do NOT repeat the original question. Each related question should be no longer than 20 words. Here is the original question:

{question}
"#
        );

        let request = ChatCompletionRequestBuilder::default()
            .messages(vec![chat::Message {
                role: "user".to_owned(),
                content: prompt,
            }])
            .build()
            .expect("Failed to create ChatCompletionRequest");

        let chat = self.chat.clone();
        let s = chat.generate(request).await;

        let s = stream! {
            let mut content = String::default();
            for await chunk in s {
                content += &chunk.choices[0].delta.content;
            }

            for line in content.lines() {
                yield remove_bullet_prefix(line).to_owned();
            }
        };

        Box::pin(s)
    }

    async fn generate_prompt(
        &self,
        snippets: &[&str],
        question: &str,
    ) -> String {
        let citations: Vec<String> = snippets
            .iter()
            .enumerate()
            .map(|(i, snippet)| format!("[[citation:{}]] {}", i, *snippet))
            .collect();
        let context = citations.join("\n\n");

        format!(
            r#"
You are a professional developer AI assistant built by Bobtail.DEV. You are given a user question, and please write clean, concise and accurate answer to the question. You will be given a set of related contexts to the question, each starting with a reference number like [[citation:x]], where x is a number. Please use the context and cite the context at the end of each sentence if applicable.

Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. Please limit to 1024 tokens. Do not give any information that is not related to the question, and do not repeat. Say "information is missing on" followed by the related topic, if the given context do not provide sufficient information.

Please cite the contexts with the reference numbers, in the format [citation:x]. If a sentence comes from multiple contexts, please list all applicable citations, like [citation:3][citation:5]. Other than code and specific names and citations, your answer must be written in the same language as the question.

Here are the set of contexts:

{context}

Remember, don't blindly repeat the contexts verbatim. When possible, give code snippet to demonstrate the answer. And here is the user question:
{question}
"#
        )
    }
}

fn remove_bullet_prefix(s: &str) -> String {
    s.trim()
        .trim_start_matches(|c: char| c == '-' || c == '*' || c == '.' || c.is_numeric())
        .trim()
        .to_owned()
}

pub fn create(
    chat: Arc<ChatService>,
    doc: Arc<dyn DocSearch>,
) -> AnswerService {
    AnswerService::new(chat, doc)
}
