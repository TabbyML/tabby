use std::sync::Arc;

use async_stream::stream;
use async_trait::async_trait;
use futures::{stream::BoxStream, StreamExt};
use serde_json::json;
use tabby_common::index::{self, corpus, doc};
use tabby_inference::Embedding;
use tantivy::doc;
use text_splitter::TextSplitter;
use tokio::task::JoinHandle;
use tracing::warn;

use crate::{
    indexer::{IndexId, TantivyDocBuilder, ToIndexId},
    IndexAttributeBuilder, Indexer,
};

pub struct SourceDocument {
    pub source_id: String,
    pub id: String,
    pub title: String,
    pub link: String,
    pub body: String,
}

impl ToIndexId for SourceDocument {
    fn to_index_id(&self) -> IndexId {
        IndexId {
            source_id: self.source_id.clone(),
            id: self.id.clone(),
        }
    }
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
    ) -> BoxStream<JoinHandle<(Vec<String>, serde_json::Value)>> {
        let embedding = self.embedding.clone();
        let chunks: Vec<_> = TextSplitter::new(CHUNK_SIZE)
            .chunks(&document.body)
            .map(|x| x.to_owned())
            .collect();

        let s = stream! {
            for chunk_text in chunks {
                let embedding = embedding.clone();
                yield tokio::spawn(async move {
                    let chunk_embedding_tokens = build_tokens(embedding.clone(), &chunk_text).await;
                    let chunk = json!({
                        doc::fields::CHUNK_TEXT: chunk_text,
                    });

                    (chunk_embedding_tokens, chunk)
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

pub fn create_web_builder(embedding: Arc<dyn Embedding>) -> TantivyDocBuilder<SourceDocument> {
    let builder = DocBuilder::new(embedding);
    TantivyDocBuilder::new(corpus::WEB, builder)
}

pub struct DocIndexer {
    builder: TantivyDocBuilder<SourceDocument>,
    indexer: Indexer,
}

pub struct WebDocument {
    pub id: String,
    pub source_id: String,
    pub link: String,
    pub title: String,
    pub body: String,
}

impl From<WebDocument> for SourceDocument {
    fn from(value: WebDocument) -> Self {
        Self {
            id: value.id,
            source_id: value.source_id,
            link: value.link,
            title: value.title,
            body: value.body,
        }
    }
}

impl DocIndexer {
    pub fn new(embedding: Arc<dyn Embedding>) -> Self {
        let builder = create_web_builder(embedding);
        let indexer = Indexer::new(corpus::WEB);
        Self { indexer, builder }
    }

    pub async fn add(&self, document: WebDocument) {
        stream! {
            let (id, s) = self.builder.build(document.into()).await;
            self.indexer.delete(&id);
            for await doc in s.buffer_unordered(std::cmp::max(std::thread::available_parallelism().unwrap().get() * 2, 32)) {
                if let Ok(Some(doc)) = doc {
                    self.indexer.add(doc).await;
                }
            }
        }.collect::<()>().await;
    }

    pub fn commit(self) {
        self.indexer.commit();
    }
}
