use std::sync::Arc;

use async_stream::stream;
use futures::{stream::BoxStream, StreamExt};
use serde::{Deserialize, Serialize};
use tabby_common::api::{
    chat::Message,
    doc::{DocSearch, DocSearchDocument},
};
use tracing::{debug, warn};
use utoipa::ToSchema;

use crate::services::chat::{ChatCompletionRequestBuilder, ChatService};

#[derive(Deserialize, ToSchema)]
#[schema(example=json!({
    "messages": [
        Message { role: "user".to_owned(), content: "What is tail recursion?".to_owned()},
    ]
}))]
pub struct AnswerRequest {
    messages: Vec<Message>,
}

#[derive(Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum AnswerResponseChunk {
    RelevantDocuments(Vec<DocSearchDocument>),
    RelevantQuestions(Vec<String>),
    AnswerDelta(String),
}
pub struct AnswerService {
    chat: Arc<ChatService>,
    doc: Arc<dyn DocSearch>,
    serper: Option<Box<dyn DocSearch>>,
}

impl AnswerService {
    fn new(chat: Arc<ChatService>, doc: Arc<dyn DocSearch>) -> Self {
        if let Ok(api_key) = std::env::var("SERPER_API_KEY") {
            debug!("Serper API key found, enabling serper...");
            let serper = Box::new(super::doc::create_serper(api_key.as_str()));
            Self {
                chat,
                doc,
                serper: Some(serper),
            }
        } else {
            Self {
                chat,
                doc,
                serper: None,
            }
        }
    }

    pub async fn answer<'a>(
        self: Arc<Self>,
        mut req: AnswerRequest,
    ) -> BoxStream<'a, AnswerResponseChunk> {
        let s = stream! {
            // 1. Collect sources given query, for now we only use the last message
            let query: &mut Message = match req.messages.last_mut() {
                Some(query) => query,
                None => {
                    warn!("No query found in the request");
                    return;
                }
            };

            // 2. Generate relevant docs from the query
            // For now we only collect from DocSearch.
            let mut hits = match self.doc.search(&query.content, 5, 0).await {
                Ok(docs) => docs.hits,
                Err(err) => {
                    warn!("Failed to search tantivy docs: {:?}", err);
                    vec![]
                }
            };

            // If serper is available, we also collect from serper
            if let Some(serper) = self.serper.as_ref() {
                let serper_hits = match serper.search(&query.content, 5, 0).await {
                    Ok(docs) => docs.hits,
                    Err(err) => {
                        warn!("Failed to search serper: {:?}", err);
                        vec![]
                    }
                };
                hits.extend(serper_hits);
            }

            yield AnswerResponseChunk::RelevantDocuments(hits.iter().map(|hit| hit.doc.clone()).collect());

            // 3. Generate relevant answers from the query
            let snippets = hits.iter().map(|hit| hit.doc.snippet.as_str()).collect::<Vec<_>>();
            yield AnswerResponseChunk::RelevantQuestions(self.generate_relevant_questions(&snippets, &query.content).await);

            // 4. Generate override prompt from the query
            query.content = self.generate_prompt(&snippets, &query.content).await;

            // 5. Generate answer from the query
            let s = self.chat.clone().generate(ChatCompletionRequestBuilder::default()
                .messages(req.messages.clone())
                .build()
                .expect("Failed to create ChatCompletionRequest"))
                .await;

            for await chunk in s {
                yield AnswerResponseChunk::AnswerDelta(chunk.choices[0].delta.content.clone());
            }
        };

        Box::pin(s)
    }

    async fn generate_relevant_questions(&self, snippets: &[&str], question: &str) -> Vec<String> {
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
            .messages(vec![Message {
                role: "user".to_owned(),
                content: prompt,
            }])
            .build()
            .expect("Failed to create ChatCompletionRequest");

        let chat = self.chat.clone();
        let s = chat.generate(request).await;

        let mut content = String::default();
        s.for_each(|chunk| {
            content += &chunk.choices[0].delta.content;
            futures::future::ready(())
        })
        .await;

        content.lines().map(remove_bullet_prefix).collect()
    }

    async fn generate_prompt(&self, snippets: &[&str], question: &str) -> String {
        let citations: Vec<String> = snippets
            .iter()
            .enumerate()
            .map(|(i, snippet)| format!("[[citation:{}]] {}", i, *snippet))
            .collect();
        let context = citations.join("\n\n");

        format!(
            r#"
You are a professional developer AI assistant. You are given a user question, and please write clean, concise and accurate answer to the question. You will be given a set of related contexts to the question, each starting with a reference number like [[citation:x]], where x is a number. Please use the context and cite the context at the end of each sentence if applicable.

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

pub fn create(chat: Arc<ChatService>, doc: Arc<dyn DocSearch>) -> AnswerService {
    AnswerService::new(chat, doc)
}
