use std::sync::Arc;

use async_stream::stream;
use async_trait::async_trait;
use futures::stream::BoxStream;
use serde_json::json;
use tabby_common::{
    config::CodeRepository,
    index::{code, corpus},
};
use tabby_inference::Embedding;
use tokio::task::JoinHandle;
use tracing::warn;

use self::intelligence::SourceCode;
use crate::{
    code::intelligence::CodeIntelligence, indexer::TantivyDocBuilder, IndexAttributeBuilder,
};

//  Modules for creating code search index.
mod index;
mod intelligence;
mod languages;
mod repository;
mod types;

#[derive(Default)]
pub struct CodeIndexer {}

impl CodeIndexer {
    pub async fn refresh(
        &mut self,
        embedding: Arc<dyn Embedding>,
        repository: &CodeRepository,
    ) -> anyhow::Result<()> {
        logkit::info!(
            "Building source code index: {}",
            repository.canonical_git_url()
        );
        repository::sync_repository(repository)?;

        index::index_repository(embedding, repository).await;
        index::garbage_collection().await;

        Ok(())
    }

    pub async fn garbage_collection(&mut self, repositories: &[CodeRepository]) {
        repository::garbage_collection(repositories);
    }
}
struct CodeBuilder {
    embedding: Option<Arc<dyn Embedding>>,
}

impl CodeBuilder {
    fn new(embedding: Option<Arc<dyn Embedding>>) -> Self {
        Self { embedding }
    }
}

#[async_trait]
impl IndexAttributeBuilder<SourceCode> for CodeBuilder {
    async fn build_attributes(&self, _source_code: &SourceCode) -> serde_json::Value {
        json!({})
    }

    async fn build_chunk_attributes(
        &self,
        source_code: &SourceCode,
    ) -> BoxStream<JoinHandle<(Vec<String>, serde_json::Value)>> {
        let text = match source_code.read_content() {
            Ok(content) => content,
            Err(e) => {
                warn!(
                    "Failed to read content of '{}': {}",
                    source_code.filepath, e
                );

                return Box::pin(futures::stream::empty());
            }
        };

        let Some(embedding) = self.embedding.clone() else {
            warn!("No embedding service found for code indexing");
            return Box::pin(futures::stream::empty());
        };

        let source_code = source_code.clone();
        let s = stream! {
            for await (start_line, body) in CodeIntelligence::chunks(&text, &source_code.language) {
                let attributes = json!({
                    code::fields::CHUNK_FILEPATH: source_code.filepath,
                    code::fields::CHUNK_GIT_URL: source_code.git_url,
                    code::fields::CHUNK_LANGUAGE: source_code.language,
                    code::fields::CHUNK_BODY:  body,
                    code::fields::CHUNK_START_LINE: start_line,
                });

                let embedding = embedding.clone();
                yield tokio::spawn(async move {
                    let tokens = build_binarize_embedding_tokens(embedding.clone(), &body).await;
                    (tokens, attributes)
                });
            }
        };

        Box::pin(s)
    }
}

async fn build_binarize_embedding_tokens(embedding: Arc<dyn Embedding>, body: &str) -> Vec<String> {
    let embedding = match embedding.embed(body).await {
        Ok(x) => x,
        Err(err) => {
            warn!("Failed to embed chunk text: {}", err);
            return Vec::new();
        }
    };

    let mut tokens = code::tokenize_code(body);
    for token in tabby_common::index::binarize_embedding(embedding.iter()) {
        tokens.push(token);
    }

    tokens
}

fn create_code_builder(embedding: Option<Arc<dyn Embedding>>) -> TantivyDocBuilder<SourceCode> {
    let builder = CodeBuilder::new(embedding);
    TantivyDocBuilder::new(corpus::CODE, builder)
}
