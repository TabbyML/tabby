use std::sync::Arc;

use anyhow::Result;
use async_stream::stream;
use async_trait::async_trait;
use futures::stream::BoxStream;
use serde_json::json;
use tabby_common::index::structured_doc::fields;
use tabby_inference::Embedding;
use text_splitter::TextSplitter;
use tokio::task::JoinHandle;

use super::{build_tokens, BuildStructuredDoc};

pub struct PageDocument {
    pub link: String,
    pub title: String,
    pub content: String,
}

#[async_trait]
impl BuildStructuredDoc for PageDocument {
    fn should_skip(&self) -> bool {
        self.content.trim().is_empty()
    }

    async fn build_attributes(&self) -> serde_json::Value {
        json!({
            fields::page::LINK: self.link,
            fields::page::TITLE: self.title,
        })
    }

    async fn build_chunk_attributes(
        &self,
        embedding: Arc<dyn Embedding>,
    ) -> BoxStream<JoinHandle<Result<(Vec<String>, serde_json::Value)>>> {
        let content = format!("{}\n\n{}", self.title, self.content);

        let chunks: Vec<_> = TextSplitter::new(2048)
            .chunks(&content)
            .map(|x| x.to_owned())
            .collect();

        let s = stream! {
            for chunk_text in chunks {
                let embedding = embedding.clone();
                yield tokio::spawn(async move {
                    let tokens = match build_tokens(embedding.clone(), &chunk_text).await {
                        Ok(tokens) => tokens,
                        Err(e) => {
                            return Err(anyhow::anyhow!("Failed to build tokens for chunk: {}", e));
                        }
                    };
                    let chunk = json!({
                        fields::page::CHUNK_CONTENT: chunk_text,
                    });

                    Ok((tokens, chunk))
                });
            }
        };

        Box::pin(s)
    }
}
