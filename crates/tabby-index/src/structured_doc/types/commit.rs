use std::sync::Arc;

use anyhow::Result;
use async_stream::stream;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::stream::BoxStream;
use serde_json::json;
use tabby_common::index::structured_doc::fields::commit;
use tabby_inference::Embedding;
use tokio::task::JoinHandle;

use super::{build_tokens, BuildStructuredDoc};

#[derive(Debug)]
pub struct CommitDocument {
    pub sha: String,
    pub message: String,
    pub author_email: String,
    pub author_at: DateTime<Utc>,
}

#[async_trait]
impl BuildStructuredDoc for CommitDocument {
    fn should_skip(&self) -> bool {
        false
    }

    async fn build_attributes(&self) -> serde_json::Value {
        json!({
            commit::SHA: self.sha,
            commit::MESSAGE: self.message,
            commit::AUTHOR_EMAIL: self.author_email,
            commit::AUTHOR_AT: self.author_at,
        })
    }

    async fn build_chunk_attributes(
        &self,
        embedding: Arc<dyn Embedding>,
    ) -> BoxStream<JoinHandle<Result<(Vec<String>, serde_json::Value)>>> {
        let s = stream! {
                let embedding = embedding.clone();
                let body = self.message.clone();
                yield tokio::spawn(async move {
                    match build_tokens(embedding.clone(), &body).await {
                        Ok(tokens) => Ok((tokens, json!({}))),
                        Err(err) => Err(err),
                    }
                });
        };

        Box::pin(s)
    }
}
