pub mod indexer;
pub mod types;

use std::{sync::Arc, vec};

use anyhow::{bail, Result};
use async_stream::stream;
use async_trait::async_trait;
use futures::stream::BoxStream;
use serde_json::json;
use tabby_common::index::{commit::fields, corpus};
use tabby_inference::Embedding;
use tokio::task::JoinHandle;
use tracing::{info_span, Instrument};
use types::CommitHistory;

use crate::IndexAttributeBuilder;

pub struct CommitHistoryBuilder {
    embedding: Arc<dyn Embedding>,
}

impl CommitHistoryBuilder {
    pub fn new(embedding: Arc<dyn Embedding>) -> Self {
        Self { embedding }
    }
}

#[async_trait]
impl IndexAttributeBuilder<CommitHistory> for CommitHistoryBuilder {
    async fn build_attributes(&self, commit: &CommitHistory) -> serde_json::Value {
        json!({
            fields::GIT_URL: commit.git_url,
            fields::SHA: commit.sha,
            fields::MESSAGE: commit.message,
            fields::AUTHOR_EMAIL: commit.author_email,
            fields::AUTHOR_AT: commit.author_at,
            fields::COMMITTER: commit.committer_email,
            fields::COMMIT_AT: commit.commit_at,
        })
    }

    async fn build_chunk_attributes<'a>(
        &self,
        commit: &'a CommitHistory,
    ) -> BoxStream<'a, JoinHandle<Result<(Vec<String>, serde_json::Value)>>> {
        let embedding = self.embedding.clone();
        let diffs = commit.diff.clone();

        let s = stream! {
            for diff in diffs.iter() {
                let attributes = json!({
                    fields::CHUNK_FILEPATH: diff.path,
                    fields::CHUNK_DIFF: diff.content,
                });

                let rewritten_body = format!("```{}\n{}\n```", diff.path, diff.content);
                let embedding = embedding.clone();
                yield tokio::spawn(async move {
                    match build_binarize_embedding_tokens(embedding.clone(), &rewritten_body).await {
                        Ok(tokens) => Ok((tokens, attributes)),
                        Err(err) => Err(err),
                    }
                });
            }
        };

        Box::pin(s)
    }
}

async fn build_binarize_embedding_tokens(
    embedding: Arc<dyn Embedding>,
    body: &str,
) -> Result<Vec<String>> {
    let embedding = match embedding
        .embed(body)
        .instrument(info_span!(
            "index_compute_embedding",
            corpus = corpus::COMMIT_HISTORY
        ))
        .await
    {
        Ok(x) => x,
        Err(err) => {
            bail!("Failed to embed commit chunk text: {}", err);
        }
    };

    let mut tokens = vec![];
    for token in tabby_common::index::binarize_embedding(embedding.iter()) {
        tokens.push(token);
    }

    Ok(tokens)
}
