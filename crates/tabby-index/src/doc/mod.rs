pub mod public;

use std::{collections::HashSet, sync::Arc};

use async_stream::stream;
use async_trait::async_trait;
use futures::stream::BoxStream;
use public::WebDocument;
use serde_json::json;
use tabby_common::index::{self, corpus, doc};
use tabby_inference::Embedding;
use tantivy::doc;
use text_splitter::TextSplitter;
use tokio::task::JoinHandle;
use tracing::warn;

use crate::{indexer::TantivyDocBuilder, IndexAttributeBuilder};

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
impl IndexAttributeBuilder<WebDocument> for DocBuilder {
    async fn build_attributes(&self, document: &WebDocument) -> serde_json::Value {
        json!({
            doc::fields::TITLE: document.title,
            doc::fields::LINK: document.link,
        })
    }

    /// This function splits the document into chunks and computes the embedding for each chunk. It then converts the embeddings
    /// into binarized tokens by thresholding on zero.
    async fn build_chunk_attributes<'a>(
        &self,
        document: &'a WebDocument,
    ) -> BoxStream<'a, JoinHandle<(Vec<String>, serde_json::Value)>> {
        let embedding = self.embedding.clone();
        let chunks: Vec<_> = TextSplitter::new(CHUNK_SIZE)
            .chunks(&document.body)
            .map(|x| x.to_owned())
            .collect();

        let title_embedding_tokens = build_tokens(embedding.clone(), &document.title).await;
        let s = stream! {
            for chunk_text in chunks {
                let title_embedding_tokens = title_embedding_tokens.clone();
                let embedding = embedding.clone();
                yield tokio::spawn(async move {
                    let chunk_embedding_tokens = build_tokens(embedding.clone(), &chunk_text).await;
                    let chunk = json!({
                        doc::fields::CHUNK_TEXT: chunk_text,
                    });

                    // Title embedding tokens are merged with chunk embedding tokens to enhance the search results.
                    let tokens = merge_tokens(vec![title_embedding_tokens, chunk_embedding_tokens]);
                    (tokens, chunk)
                });
            }
        };

        Box::pin(s)
    }
}

async fn build_tokens(embedding: Arc<dyn Embedding>, text: &str) -> Vec<String> {
    let embedding = match embedding.embed(text).await {
        Ok(embedding) => embedding,
        Err(err) => {
            warn!("Failed to embed chunk text: {}", err);
            return vec![];
        }
    };

    let mut chunk_embedding_tokens = vec![];
    for token in index::binarize_embedding(embedding.iter()) {
        chunk_embedding_tokens.push(token);
    }

    chunk_embedding_tokens
}

fn create_web_builder(embedding: Arc<dyn Embedding>) -> TantivyDocBuilder<WebDocument> {
    let builder = DocBuilder::new(embedding);
    TantivyDocBuilder::new(corpus::WEB, builder)
}

pub fn merge_tokens(tokens: Vec<Vec<String>>) -> Vec<String> {
    let tokens = tokens.into_iter().flatten().collect::<HashSet<_>>();
    tokens.into_iter().collect()
}
