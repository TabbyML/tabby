use std::sync::Arc;

use anyhow::Result;
use async_stream::stream;
use async_trait::async_trait;
use futures::stream::BoxStream;
use serde_json::json;
use tabby_common::index::structured_doc::fields;
use tabby_inference::Embedding;
use tokio::task::JoinHandle;

use super::{build_tokens, BuildStructuredDoc};

pub struct IssueDocument {
    pub link: String,
    pub title: String,
    pub author_email: Option<String>,
    pub body: String,
    pub closed: bool,
}

#[async_trait]
impl BuildStructuredDoc for IssueDocument {
    fn should_skip(&self) -> bool {
        false
    }

    async fn build_attributes(&self) -> serde_json::Value {
        json!({
            fields::issue::LINK: self.link,
            fields::issue::TITLE: self.title,
            fields::issue::AUTHOR_EMAIL: self.author_email,
            fields::issue::BODY: self.body,
            fields::issue::CLOSED: self.closed,
        })
    }

    async fn build_chunk_attributes(
        &self,
        embedding: Arc<dyn Embedding>,
    ) -> BoxStream<JoinHandle<Result<(Vec<String>, serde_json::Value)>>> {
        let text = format!("{}\n\n{}", self.title, self.body);
        let s = stream! {
            yield tokio::spawn(async move {
                let tokens = match build_tokens(embedding, &text).await{
                    Ok(tokens) => tokens,
                    Err(e) => {
                        return Err(anyhow::anyhow!("Failed to build tokens for text: {}", e));
                    }
                };
                let chunk_attributes = json!({});
                Ok((tokens, chunk_attributes))
            })
        };

        Box::pin(s)
    }
}
