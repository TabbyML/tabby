use std::sync::Arc;

use async_stream::stream;
use async_trait::async_trait;
use futures::stream::BoxStream;
use serde_json::json;
use tabby_common::{
    config::RepositoryConfig,
    index::{code, corpus},
};
use tabby_inference::Embedding;
use tracing::warn;

use self::intelligence::SourceCode;
use crate::{code::intelligence::CodeIntelligence, IndexAttributeBuilder, Indexer};

//  Modules for creating code search index.
mod index;
mod intelligence;
mod languages;
mod repository;
mod types;

#[derive(Default)]
pub struct CodeIndexer {}

impl CodeIndexer {
    pub async fn refresh(&mut self, embedding: Arc<dyn Embedding>, repository: &RepositoryConfig) {
        logkit::info!(
            "Building source code index: {}",
            repository.canonical_git_url()
        );
        repository::sync_repository(repository);

        index::index_repository(embedding, repository).await;
        index::garbage_collection().await;
    }

    pub async fn garbage_collection(&mut self, repositories: &[RepositoryConfig]) {
        repository::garbage_collection(repositories);
    }
}
struct CodeBuilder {
    embedding: Option<Arc<dyn Embedding>>,
    intelligence: CodeIntelligence,
}

impl CodeBuilder {
    fn new(embedding: Option<Arc<dyn Embedding>>) -> Self {
        Self {
            embedding,
            intelligence: CodeIntelligence::default(),
        }
    }
}

#[async_trait]
impl IndexAttributeBuilder<SourceCode> for CodeBuilder {
    async fn build_id(&self, source_code: &SourceCode) -> String {
        source_code.id.clone()
    }

    async fn build_source(&self, source_code: &SourceCode) -> Option<String> {
        None
    }

    async fn build_attributes(&self, _source_code: &SourceCode) -> serde_json::Value {
        json!({})
    }

    async fn build_chunk_attributes(
        &self,
        source_code: &SourceCode,
    ) -> BoxStream<(Vec<String>, serde_json::Value)> {
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
            for (start_line, body) in self.intelligence.chunks(&text, &source_code.language) {
                let embedding = match embedding.embed(&body).await {
                    Ok(x) => x,
                    Err(err) => {
                        warn!("Failed to embed chunk text: {}", err);
                        continue;
                    }
                };

                let mut tokens = code::tokenize_code(&body);
                for token in tabby_common::index::binarize_embedding(embedding.iter()) {
                    tokens.push(token);
                }
                yield (tokens, json!({
                    code::fields::CHUNK_FILEPATH: source_code.filepath,
                    code::fields::CHUNK_GIT_URL: source_code.git_url,
                    code::fields::CHUNK_LANGUAGE: source_code.language,
                    code::fields::CHUNK_BODY:  body,
                    code::fields::CHUNK_START_LINE: start_line,
                }));
            }
        };

        Box::pin(s)
    }
}

fn create_code_index(embedding: Option<Arc<dyn Embedding>>) -> Indexer<SourceCode> {
    let builder = CodeBuilder::new(embedding);
    Indexer::new(corpus::CODE, builder)
}
