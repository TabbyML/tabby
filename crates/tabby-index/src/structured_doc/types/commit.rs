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
    pub git_url: String,
    pub sha: String,
    pub message: String,
    pub author_email: String,
    pub author_at: DateTime<Utc>,
    pub committer_email: String,
    pub commit_at: DateTime<Utc>,

    pub diff: Vec<CommitDiff>,
}

#[derive(Debug, Clone)]
pub struct CommitDiff {
    pub path: String,
    pub content: String,
}

#[async_trait]
impl BuildStructuredDoc for CommitDocument {
    fn should_skip(&self) -> bool {
        false
    }

    async fn build_attributes(&self) -> serde_json::Value {
        json!({
            commit::GIT_URL: self.git_url,
            commit::SHA: self.sha,
            commit::MESSAGE: self.message,
            commit::AUTHOR_EMAIL: self.author_email,
            commit::AUTHOR_AT: self.author_at,
            commit::COMMITTER_EMAIL: self.committer_email,
            commit::COMMIT_AT: self.commit_at,
        })
    }

    async fn build_chunk_attributes(
        &self,
        embedding: Arc<dyn Embedding>,
    ) -> BoxStream<JoinHandle<Result<(Vec<String>, serde_json::Value)>>> {
        let diffs = self.diff.clone();

        let s = stream! {
            for diff in diffs.iter() {
                let attributes = json!({
                    commit::CHUNK_FILEPATH: diff.path,
                    commit::CHUNK_DIFF: diff.content,
                });

                let rewritten_body = format!(r#"Commit Message:
```
{}
```
Changed file: {}
Changed Content:
```
{}
```
"#, self.message, diff.path, diff.content);
                let embedding = embedding.clone();
                yield tokio::spawn(async move {
                    match build_tokens(embedding.clone(), &rewritten_body).await {
                        Ok(tokens) => Ok((tokens, attributes)),
                        Err(err) => Err(err),
                    }
                });
            }
        };

        Box::pin(s)
    }
}
