pub mod indexer;
pub mod types;

use std::{pin::pin, sync::Arc, vec};

use anyhow::{bail, Result};
use async_stream::stream;
use async_trait::async_trait;
use futures::{stream::BoxStream, StreamExt};
use git2::{Repository, Sort};
use serde_json::json;
use tabby_common::{
    config::CodeRepository,
    index::{commit::fields, corpus},
};
use tabby_inference::Embedding;
use tokio::{sync::oneshot, task::JoinHandle};
use tracing::warn;
use tracing::{info_span, Instrument};
use types::CommitHistory;

use crate::{
    indexer::{Indexer, TantivyDocBuilder},
    IndexAttributeBuilder,
};

pub struct CommitHistoryBuilder {
    embedding: Arc<dyn Embedding>,
}

impl CommitHistoryBuilder {
    pub fn new(embedding: Arc<dyn Embedding>) -> Self {
        Self { embedding }
    }

    pub async fn garbage_collection(&self) {
        let index = Indexer::new(corpus::COMMIT_HISTORY);
        stream! {
            let mut num_to_keep = 0;
            let mut num_to_delete = 0;

            // for await id in index.iter_ids() {
            //     let Some(source_file_id) = SourceCode::source_file_id_from_id(&id) else {
            //         warn!("Failed to extract source file id from index id: {id}");
            //         num_to_delete += 1;
            //         index.delete(&id);
            //         continue;
            //     };

            //     if CodeIntelligence::check_source_file_id_matched(source_file_id) {
            //         num_to_keep += 1;
            //     } else {
            //         num_to_delete += 1;
            //         index.delete(&id);
            //     }
            // }

            logkit::info!("Finished garbage collection for code index: {num_to_keep} items kept, {num_to_delete} items removed");
            index.commit();
        }.collect::<()>().await;
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
            fields::COMMITTER: document.committer_email,
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
