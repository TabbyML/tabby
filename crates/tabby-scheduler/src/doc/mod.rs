use std::sync::Arc;

use async_stream::stream;
use async_trait::async_trait;
use futures::stream::BoxStream;
use serde_json::json;
use tabby_common::index::{self, corpus, doc};
use tabby_inference::Embedding;
use tantivy::doc;
use text_splitter::TextSplitter;
use tracing::warn;

use crate::{IndexAttributeBuilder, Indexer};

pub struct SourceDocument {
    pub id: String,
    pub title: String,
    pub link: String,
    pub body: String,
}

const CHUNK_SIZE: usize = 2048;

pub struct DocBuilder {
    embedding: Arc<dyn Embedding>,
}

impl DocBuilder {
    fn new(embedding: Arc<dyn Embedding>) -> Self {
        Self { embedding }
    }
}

#[async_trait]
impl IndexAttributeBuilder<SourceDocument> for DocBuilder {
    async fn build_id(&self, document: &SourceDocument) -> String {
        document.id.clone()
    }

    async fn build_attributes(&self, document: &SourceDocument) -> serde_json::Value {
        json!({
            doc::fields::TITLE: document.title,
            doc::fields::LINK: document.link,
        })
    }

    /// This function splits the document into chunks and computes the embedding for each chunk. It then converts the embeddings
    /// into binarized tokens by thresholding on zero.
    async fn build_chunk_attributes(
        &self,
        document: &SourceDocument,
    ) -> BoxStream<(Vec<String>, serde_json::Value)> {
        let embedding = self.embedding.clone();
        let chunks: Vec<_> = TextSplitter::new(CHUNK_SIZE)
            .chunks(&document.body)
            .map(|x| x.to_owned())
            .collect();

        let s = stream! {
            for chunk_text in chunks {
                let embedding = match embedding.embed(&chunk_text).await {
                    Ok(embedding) => embedding,
                    Err(err) => {
                        warn!("Failed to embed chunk text: {}", err);
                        continue;
                    }
                };

                let mut chunk_embedding_tokens = vec![];
                for token in index::binarize_embedding(embedding.iter()) {
                    chunk_embedding_tokens.push(token);
                }

                let chunk = json!({
                    doc::fields::CHUNK_TEXT: chunk_text,
                });

                yield (chunk_embedding_tokens, chunk)
            }
        };

        Box::pin(s)
    }
}

pub fn create_web_index(embedding: Arc<dyn Embedding>) -> Indexer<SourceDocument> {
    let builder = DocBuilder::new(embedding);
    Indexer::new(corpus::WEB, builder)
}
