use std::sync::Arc;

use async_stream::stream;
use async_trait::async_trait;
use futures::{stream::BoxStream, StreamExt};
use serde_json::json;
use tabby_common::index::{webdoc, DocSearchSchema};
use tabby_inference::Embedding;
use tantivy::doc;
use text_splitter::TextSplitter;
use tracing::warn;

use crate::{DocIndex, DocumentBuilder};

pub struct SourceDocument {
    pub id: String,
    pub title: String,
    pub link: String,
    pub body: String,
}

const CHUNK_SIZE: usize = 2048;

pub struct WebBuilder {
    embedding: Arc<dyn Embedding>,
}

impl WebBuilder {
    pub fn new(embedding: Arc<dyn Embedding>) -> Self {
        Self { embedding }
    }
}

#[async_trait]
impl DocumentBuilder<SourceDocument> for WebBuilder {
    async fn build_id(&self, document: &SourceDocument) -> String {
        document.id.clone()
    }

    async fn build_attributes(&self, document: &SourceDocument) -> serde_json::Value {
        json!({
            webdoc::fields::TITLE: document.title,
            webdoc::fields::LINK: document.link,
        })
    }

    /// This function splits the document into chunks and computes the embedding for each chunk. It then converts the embeddings
    /// into binarized tokens by thresholding on zero.
    async fn build_chunk_attributes(
        &self,
        document: &SourceDocument,
    ) -> BoxStream<serde_json::Value> {
        let splitter = TextSplitter::default().with_trim_chunks(true);
        let embedding = self.embedding.clone();
        let content = document.body.clone();

        let s = stream! {
            for chunk_text in splitter.chunks(&content, CHUNK_SIZE) {
                let embedding = match embedding.embed(chunk_text).await {
                    Ok(embedding) => embedding,
                    Err(err) => {
                        warn!("Failed to embed chunk text: {}", err);
                        continue;
                    }
                };

                let mut chunk_embedding_tokens = vec![];
                for token in DocSearchSchema::binarize_embedding(embedding.iter()) {
                    chunk_embedding_tokens.push(token);
                }

                let chunk = json!({
                        // FIXME: tokenize chunk text
                        webdoc::fields::CHUNK_TEXT: chunk_text,
                        webdoc::fields::CHUNK_EMBEDDING: chunk_embedding_tokens,
                });

                yield chunk
            }
        };

        Box::pin(s)
    }
}

pub fn create_web_index(embedding: Arc<dyn Embedding>) -> DocIndex<SourceDocument> {
    let builder = WebBuilder::new(embedding);
    DocIndex::new(builder)
}
