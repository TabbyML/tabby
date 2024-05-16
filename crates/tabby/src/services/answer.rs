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
use tabby_common::api::{
    chat,
    code::{CodeSearch, CodeSearchDocument},
    doc::{DocSearch, DocSearchDocument},
};
use tabby_inference::ChatCompletionStream;
use tracing::{instrument, warn};
use utoipa::{IntoParams, ToSchema};

use crate::services::chat::ChatService;

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
pub enum AnswerResponseChunk {
    RelevantCode(CodeSearchDocument),
    RelevantDoc(DocSearchDocument),
    RelevantQuestion(String),
    Answer(String),
}
pub struct AnswerService {
    chat: Arc<ChatService>,
    code: Arc<dyn CodeSearch>,
    doc: Arc<dyn DocSearch>,
}

impl AnswerService {
    fn new(chat: Arc<ChatService>, code: Arc<dyn CodeSearch>, doc: Arc<dyn DocSearch>) -> Self {
        Self { chat, code, doc }
    }

    pub async fn answer<'a>(
        self: Arc<Self>,
        mut req: AnswerRequest,
    ) -> BoxStream<'a, AnswerResponseChunk> {
        let s = stream! {
            // 1. Collect sources given query, for now we only use the last message
            let query = match req.messages.last() {
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

            for hit in docs.hits {
                yield AnswerResponseChunk::RelevantDoc(hit.doc);
            }

            // 3. Generate relevant answers from the query
            // TBD.

            // 4. Generate answer from the query
        };

        Box::pin(s)
    }
}

pub fn create(
    chat: Arc<ChatService>,
    code: Arc<dyn CodeSearch>,
    doc: Arc<dyn DocSearch>,
) -> AnswerService {
    AnswerService::new(chat, code, doc)
}
