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

pub struct IngestedDocument {
    // the link of the document is optional,
    // so we use source/doc_id as the unique identifier.
    pub id: String,
    pub title: String,
    pub body: String,
    pub link: Option<String>,
}

#[async_trait]
impl BuildStructuredDoc for IngestedDocument {
    fn should_skip(&self) -> bool {
        self.body.trim().is_empty()
    }

    async fn build_attributes(&self) -> serde_json::Value {
        let mut attr = json!({
            fields::ingested::TITLE: self.title,
        });
        if let Some(link) = &self.link {
            attr.as_object_mut()
                .unwrap()
                .insert(fields::ingested::LINK.to_string(), json!(link));
        };

        attr
    }

    async fn build_chunk_attributes(
        &self,
        embedding: Arc<dyn Embedding>,
    ) -> BoxStream<'life0, JoinHandle<Result<(Vec<String>, serde_json::Value)>>> {
        let content = format!("{}\n\n{}", self.title, self.body);

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
                        fields::ingested::CHUNK_BODY: chunk_text,
                    });

                    Ok((tokens, chunk))
                });
            }
        };

        Box::pin(s)
    }
}
