use std::sync::Arc;

use async_stream::stream;
use async_trait::async_trait;
use futures::stream::BoxStream;
use serde_json::json;
use tabby_common::index::structured_doc::fields;
use tabby_inference::Embedding;
use tokio::task::JoinHandle;

use super::{build_tokens, BuildStructuredDoc};

pub struct PullRequest {
    pub link: String,
    pub title: String,
    pub body: String,
    pub patch: String,
    pub state: String,
}

#[async_trait]
impl BuildStructuredDoc for PullRequest {
    fn should_skip(&self) -> bool {
        false
    }

    async fn build_attributes(&self) -> serde_json::Value {
        json!({
            fields::pull::LINK: self.link,
            fields::pull::TITLE: self.title,
            fields::pull::BODY: self.body,
            fields::pull::PATCH: self.patch,
            fields::pull::STATE: self.state,
        })
    }

    async fn build_chunk_attributes(
        &self,
        embedding: Arc<dyn Embedding>,
    ) -> BoxStream<JoinHandle<(Vec<String>, serde_json::Value)>> {
        let text = format!("{}\n\n{}\n\n{}", self.title, self.body, self.patch);
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
