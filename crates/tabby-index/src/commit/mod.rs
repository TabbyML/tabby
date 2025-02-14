mod types;

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

use crate::{indexer::TantivyDocBuilder, IndexAttributeBuilder};

fn create_commit_history_builder(
    embedding: Arc<dyn Embedding>,
) -> TantivyDocBuilder<CommitHistory> {
    let builder = CommitHistoryBuilder::new(embedding);
    TantivyDocBuilder::new(corpus::COMMIT_HISTORY, builder)
}

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
    async fn build_attributes(&self, document: &CommitHistory) -> serde_json::Value {
        json!({
            fields::GIT_URL: document.git_url,
            fields::SHA: document.sha,
            fields::MESSAGE: document.message,
            fields::AUTHOR_EMAIL: document.author_email,
            fields::AUTHOR_AT: document.author_at,
            fields::COMMITTER: document.committer,
            fields::COMMIT_AT: document.commit_at,
        })
    }

    async fn build_chunk_attributes<'a>(
        &self,
        document: &'a CommitHistory,
    ) -> BoxStream<'a, JoinHandle<Result<(Vec<String>, serde_json::Value)>>> {
        let embedding = self.embedding.clone();
        let diff = document.diff.clone();

        let s = stream! {
            if let Some(diff) = diff {
                for (file, diff) in parse_diff(diff.as_ref()) {
                    let attributes = json!({
                        fields::CHUNK_FILEPATH: file,
                        fields::CHUNK_DIFF: diff,
                    });

                    let rewritten_body = format!("```{}\n{}\n```", file, diff);
                    let embedding = embedding.clone();
                    yield tokio::spawn(async move {
                        match build_binarize_embedding_tokens(embedding.clone(), &rewritten_body).await {
                            Ok(tokens) => Ok((tokens, attributes)),
                            Err(err) => Err(err),
                        }
                    });
                }
            }
        };

        Box::pin(s)
    }
}

//TODO(kweizh): parse diff
fn parse_diff(_diff: &str) -> Vec<(String, String)> {
    vec![("file".to_string(), "diff".to_string())]
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

    //TODO(kweizh): tokenize commit diff?
    let mut tokens = vec![];
    for token in tabby_common::index::binarize_embedding(embedding.iter()) {
        tokens.push(token);
    }

    Ok(tokens)
}
