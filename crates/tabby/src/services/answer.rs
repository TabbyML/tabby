use std::sync::Arc;

use async_stream::stream;
use futures::{stream::BoxStream, StreamExt};
use serde::{Deserialize, Serialize};
use tabby_common::api::{
    chat::Message,
    code::{CodeSearch, CodeSearchDocument, CodeSearchError, CodeSearchQuery},
    doc::{DocSearch, DocSearchDocument, DocSearchError},
};
use tracing::{debug, warn};
use utoipa::ToSchema;

use crate::services::chat::{ChatCompletionRequestBuilder, ChatService};

#[derive(Deserialize, ToSchema)]
#[schema(example=json!({
    "messages": [
        Message { role: "user".to_owned(), content: "What is tail recursion?".to_owned()},
    ],
}))]
pub struct AnswerRequest {
    messages: Vec<Message>,

    #[serde(default)]
    code_query: Option<CodeSearchQuery>,

    #[serde(default)]
    doc_query: bool,

    #[serde(default)]
    generate_relevant_questions: bool,
}

#[derive(Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum AnswerResponseChunk {
    RelevantCode(Vec<CodeSearchDocument>),
    RelevantDocuments(Vec<DocSearchDocument>),
    RelevantQuestions(Vec<String>),
    AnswerDelta(String),
}
pub struct AnswerService {
    chat: Arc<ChatService>,
    code: Arc<dyn CodeSearch>,
    doc: Option<Arc<dyn DocSearch>>,
    serper: Option<Box<dyn DocSearch>>,
}

// FIXME(meng): make this configurable.
const RELEVANT_CODE_THRESHOLD: f32 = 4.4;
const PRESENCE_PENALTY: f32 = 0.5;

impl AnswerService {
    fn new(
        chat: Arc<ChatService>,
        code: Arc<dyn CodeSearch>,
        doc: Option<Arc<dyn DocSearch>>,
    ) -> Self {
        let serper: Option<Box<dyn DocSearch>> =
            if let Ok(api_key) = std::env::var("SERPER_API_KEY") {
                debug!("Serper API key found, enabling serper...");
                Some(Box::new(super::doc::create_serper(api_key.as_str())))
            } else {
                None
            };
        Self {
            chat,
            code,
            doc,
            serper,
        }
    }

    pub async fn answer<'a>(
        self: Arc<Self>,
        mut req: AnswerRequest,
    ) -> BoxStream<'a, AnswerResponseChunk> {
        let s = stream! {
            // 0. Collect sources given query, for now we only use the last message
            let query: &mut Message = match req.messages.last_mut() {
                Some(query) => query,
                None => {
                    warn!("No query found in the request");
                    return;
                }
            };

            // 1. Collect relevant code if needed.
            let relevant_code = if let Some(code_query)  = req.code_query  {
                self.override_query_with_code_query(query, &code_query).await;
                self.collect_relevant_code(code_query).await
            } else {
                vec![]
            };

            if !relevant_code.is_empty() {
                yield AnswerResponseChunk::RelevantCode(relevant_code.clone());
            }


            // 2. Collect relevant docs if needed.
            let relevant_docs = if req.doc_query {
                self.collect_relevant_docs(&query.content).await
            } else {
                vec![]
            };

            if !relevant_docs.is_empty() {
                yield AnswerResponseChunk::RelevantDocuments(relevant_docs.clone());
            }

            if !relevant_code.is_empty() || !relevant_docs.is_empty() {
                if req.generate_relevant_questions {
                    // 3. Generate relevant questions from the query
                    let relevant_questions = self.generate_relevant_questions(&relevant_code, &relevant_docs, &query.content).await;
                    yield AnswerResponseChunk::RelevantQuestions(relevant_questions);
                }


                // 4. Generate override prompt from the query
                query.content = self.generate_prompt(&relevant_code, &relevant_docs, &query.content).await;
            }


            // 5. Generate answer from the query
            let s = self.chat.clone().generate(ChatCompletionRequestBuilder::default()
                .messages(req.messages.clone())
                .presence_penalty(Some(PRESENCE_PENALTY))
                .build()
                .expect("Failed to create ChatCompletionRequest"))
                .await;

            for await chunk in s {
                yield AnswerResponseChunk::AnswerDelta(chunk.choices[0].delta.content.clone());
            }
        };

        Box::pin(s)
    }

    async fn collect_relevant_code(&self, query: CodeSearchQuery) -> Vec<CodeSearchDocument> {
        let hits = match self.code.search_in_language(query, 5, 0).await {
            Ok(docs) => docs.hits,
            Err(err) => {
                if let CodeSearchError::NotReady = err {
                    debug!("Code search is not ready yet");
                } else {
                    warn!("Failed to search code: {:?}", err);
                }
                vec![]
            }
        };

        hits.into_iter()
            .inspect(|hit| {
                debug!(
                    "Code search hit: {:?}, score {:?}",
                    hit.doc.filepath, hit.score
                )
            })
            .filter(|hit| hit.score > RELEVANT_CODE_THRESHOLD)
            .map(|hit| hit.doc)
            .collect()
    }

    async fn collect_relevant_docs(&self, query: &str) -> Vec<DocSearchDocument> {
        // 1. Collect relevant docs from the tantivy doc search.
        let mut hits = vec![];
        if let Some(doc) = self.doc.as_ref() {
            let doc_hits = match doc.search(query, 5, 0).await {
                Ok(docs) => docs.hits,
                Err(err) => {
                    if let DocSearchError::NotReady = err {
                        debug!("Doc search is not ready yet");
                    } else {
                        warn!("Failed to search doc: {:?}", err);
                    }
                    vec![]
                }
            };
            hits.extend(doc_hits);
        }

        // 2. If serper is available, we also collect from serper
        if let Some(serper) = self.serper.as_ref() {
            let serper_hits = match serper.search(query, 5, 0).await {
                Ok(docs) => docs.hits,
                Err(err) => {
                    warn!("Failed to search serper: {:?}", err);
                    vec![]
                }
            };
            hits.extend(serper_hits);
        }

        hits.into_iter().map(|hit| hit.doc).collect()
    }

    async fn generate_relevant_questions(
        &self,
        relevant_code: &[CodeSearchDocument],
        relevant_docs: &[DocSearchDocument],
        question: &str,
    ) -> Vec<String> {
        let snippets: Vec<String> = relevant_code
            .iter()
            .map(|doc| {
                format!(
                    "```{} title=\"{}\"\n{}\n```",
                    doc.language, doc.filepath, doc.body
                )
            })
            .chain(relevant_docs.iter().map(|doc| doc.snippet.to_owned()))
            .collect();

        let context: String = snippets.join("\n\n");
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

    async fn override_query_with_code_query(
        &self,
        query: &mut Message,
        code_query: &CodeSearchQuery,
    ) {
        query.content = format!(
            "{}\n\n```{}\n{}\n```",
            query.content, code_query.language, code_query.content
        )
    }

    async fn generate_prompt(
        &self,
        relevant_code: &[CodeSearchDocument],
        relevant_docs: &[DocSearchDocument],
        question: &str,
    ) -> String {
        let snippets: Vec<String> = relevant_code
            .iter()
            .map(|doc| {
                format!(
                    "```{} title=\"{}\"\n{}\n```",
                    doc.language, doc.filepath, doc.body
                )
            })
            .chain(relevant_docs.iter().map(|doc| doc.snippet.to_owned()))
            .collect();

        let citations: Vec<String> = snippets
            .iter()
            .enumerate()
            .map(|(i, snippet)| format!("[[citation:{}]]\n{}", i + 1, *snippet))
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

pub fn create(
    chat: Arc<ChatService>,
    code: Arc<dyn CodeSearch>,
    doc: Option<Arc<dyn DocSearch>>,
) -> AnswerService {
    AnswerService::new(chat, code, doc)
}
