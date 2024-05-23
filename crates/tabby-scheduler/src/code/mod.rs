use async_stream::stream;
use async_trait::async_trait;
use futures::stream::BoxStream;
use serde_json::json;
use tabby_common::{config::RepositoryConfig, index::code};
use tracing::{info, warn};

use self::intelligence::SourceCode;
use crate::{code::intelligence::CodeIntelligence, IndexAttributeBuilder, Indexer};

///  Module for creating code search index.
mod cache;
mod index;
mod intelligence;
mod languages;
mod repository;
mod types;

#[derive(Default)]
pub struct CodeIndexer {
    is_dirty: bool,
}

impl CodeIndexer {
    pub async fn refresh(&mut self, repository: &RepositoryConfig) {
        self.is_dirty = true;

        info!("Refreshing repository: {}", repository.canonical_git_url());
        repository::sync_repository(repository);

        let mut cache = cache::CacheStore::new(tabby_common::path::cache_dir());
        index::index_repository(&mut cache, repository).await;
    }

    pub fn garbage_collection(&mut self, repositories: &[RepositoryConfig]) {
        self.is_dirty = false;
        let mut cache = cache::CacheStore::new(tabby_common::path::cache_dir());
        cache.garbage_collection_for_source_files();
        index::garbage_collection(&mut cache);
        repository::garbage_collection(repositories);
    }
}

struct KeyedSourceCode {
    key: String,
    code: SourceCode,
}

struct CodeBuilder;

#[async_trait]
impl IndexAttributeBuilder<KeyedSourceCode> for CodeBuilder {
    fn format_id(&self, id: &str) -> String {
        format!("code:{}", id)
    }

    async fn build_id(&self, source_code: &KeyedSourceCode) -> String {
        self.format_id(&source_code.key)
    }

    async fn build_attributes(&self, _source_code: &KeyedSourceCode) -> serde_json::Value {
        json!({})
    }

    async fn build_chunk_attributes(
        &self,
        source_code: &KeyedSourceCode,
    ) -> BoxStream<(Vec<String>, serde_json::Value)> {
        let source_code = &source_code.code;
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

        let source_code = source_code.clone();
        let s = stream! {
            let intelligence = CodeIntelligence::default();
            for (start_line, body) in intelligence.chunks(&text) {
                let tokens = code::tokenize_code(body);
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

fn create_code_index() -> Indexer<KeyedSourceCode> {
    let builder = CodeBuilder;
    Indexer::new(builder)
}
