use std::{collections::HashSet, sync::Arc};

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

pub struct WebDocument {
    pub link: String,
    pub title: String,
    pub body: String,
}

#[async_trait]
impl BuildStructuredDoc for WebDocument {
    fn should_skip(&self) -> bool {
        self.body.trim().is_empty()
    }

    async fn build_attributes(&self) -> serde_json::Value {
        json!({
            fields::web::TITLE: self.title,
            fields::web::LINK: self.link,
        })
    }

    async fn build_chunk_attributes(
        &self,
        embedding: Arc<dyn Embedding>,
    ) -> BoxStream<JoinHandle<Result<(Vec<String>, serde_json::Value)>>> {
        let chunks: Vec<_> = TextSplitter::new(2048)
            .chunks(&self.body)
            .map(|x| x.to_owned())
            .collect();

        let title_embedding_tokens = match build_tokens(embedding.clone(), &self.title).await {
            Ok(tokens) => tokens,
            Err(e) => {
                return Box::pin(stream! {
                    yield tokio::spawn(async move {
                        Err(anyhow::anyhow!("Failed to build tokens for title: {}", e))
                    });
                });
            }
        };
        let s = stream! {
            for chunk_text in chunks {
                let title_embedding_tokens = title_embedding_tokens.clone();
                let embedding = embedding.clone();
                yield tokio::spawn(async move {
                    let chunk_embedding_tokens = match build_tokens(embedding.clone(), &chunk_text).await {
                        Ok(tokens) => tokens,
                        Err(e) => {
                            return Err(anyhow::anyhow!("Failed to build tokens for chunk: {}", e));
                        }
                    };
                    let chunk = json!({
                        fields::web::CHUNK_TEXT: chunk_text,
                    });

                    // Title embedding tokens are merged with chunk embedding tokens to enhance the search results.
                    let tokens = merge_tokens(vec![title_embedding_tokens, chunk_embedding_tokens]);
                    Ok((tokens, chunk))
                });
            }
        };

        Box::pin(s)
    }
}

pub fn merge_tokens(tokens: Vec<Vec<String>>) -> Vec<String> {
    let tokens = tokens.into_iter().flatten().collect::<HashSet<_>>();
    tokens.into_iter().collect()
}
