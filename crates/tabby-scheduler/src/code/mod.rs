use async_stream::stream;
use async_trait::async_trait;
use futures::stream::BoxStream;
use serde_json::json;
use tabby_common::{
    config::RepositoryConfig,
    index::{webcode, CodeSearchSchema},
};
use tracing::{info, warn};

use self::{cache::SourceFileKey, intelligence::SourceCode};
use crate::{code::intelligence::CodeIntelligence, DocIndex, DocumentBuilder};

///  Module for creating code search index.
mod cache;
mod index;
mod intelligence;
mod languages;
mod repository;
mod types;

#[derive(Default)]
pub struct CodeIndex {
    is_dirty: bool,
}

impl CodeIndex {
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

struct CodeBuilder;

#[async_trait]
impl DocumentBuilder<SourceCode> for CodeBuilder {
    fn format_id(&self, id: &str) -> String {
        format!("code:{}", id)
    }

    async fn build_id(&self, source_code: &SourceCode) -> String {
        let path = source_code.absolute_path();
        let id = SourceFileKey::try_from(path.as_path())
            .expect("Failed to build ID from path")
            .to_string();
        self.format_id(&id)
    }

    async fn build_attributes(&self, _source_code: &SourceCode) -> serde_json::Value {
        json!({})
    }

    async fn build_chunk_attributes(
        &self,
        source_file: &SourceCode,
    ) -> BoxStream<serde_json::Value> {
        let text = match source_file.read_content() {
            Ok(content) => content,
            Err(e) => {
                warn!(
                    "Failed to read content of '{}': {}",
                    source_file.filepath, e
                );

                return Box::pin(futures::stream::empty());
            }
        };

        let source_file = source_file.clone();
        let s = stream! {
            let intelligence = CodeIntelligence::default();
            for (start_line, body) in intelligence.chunks(&text) {
                yield json!({
                    webcode::fields::CHUNK_FILEPATH: source_file.filepath,
                    webcode::fields::CHUNK_GIT_URL: source_file.git_url,
                    webcode::fields::CHUNK_LANGUAGE: source_file.language,
                    webcode::fields::CHUNK_TOKENIZED_BODY:  CodeSearchSchema::tokenize_code(body),
                    webcode::fields::CHUNK_BODY:  body,
                    webcode::fields::CHUNK_START_LINE: start_line,
                });
            }
        };

        Box::pin(s)
    }
}

fn create_code_index() -> DocIndex<SourceCode> {
    let builder = CodeBuilder;
    DocIndex::new(builder)
}
