use std::sync::Arc;

use async_stream::stream;
use async_trait::async_trait;
use futures::stream::BoxStream;
use serde_json::json;
use tabby_common::index::structured_doc::fields;
use tabby_inference::Embedding;
use tokio::task::JoinHandle;

use super::{build_tokens, BuildStructuredDoc};

pub struct PullDocument {
    pub link: String,
    pub title: String,
    pub body: String,

    /// The diff represents the code changes in this PR,
    /// including metadata, affected line ranges, and added (+) or removed (-) lines.
    /// For more details on the diff format, refer to:
    /// https://git-scm.com/docs/diff-format#_combined_diff_format
    pub diff: String,
    pub merged: bool,
}

#[async_trait]
impl BuildStructuredDoc for PullDocument {
    fn should_skip(&self) -> bool {
        false
    }

    async fn build_attributes(&self) -> serde_json::Value {
        json!({
            fields::pull::LINK: self.link,
            fields::pull::TITLE: self.title,
            fields::pull::BODY: self.body,
            fields::pull::DIFF: self.diff,
            fields::pull::MERGED: self.merged,
        })
    }

    async fn build_chunk_attributes(
        &self,
        embedding: Arc<dyn Embedding>,
    ) -> BoxStream<JoinHandle<(Vec<String>, serde_json::Value)>> {
        // currently not indexing the diff
        let text = format!("{}\n\n{}", self.title, self.body);
        let s = stream! {
            yield tokio::spawn(async move {
                let tokens = build_tokens(embedding, &text).await;
                let chunk_attributes = json!({});
                (tokens, chunk_attributes)
            })
        };

        Box::pin(s)
    }
}
