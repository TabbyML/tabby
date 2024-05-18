use tabby_common::config::RepositoryConfig;
use tracing::{info, warn};

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
    pub fn refresh(&mut self, repository: &RepositoryConfig) {
        self.is_dirty = true;

        info!("Refreshing repository: {}", repository.canonical_git_url());
        repository::sync_repository(repository);

        let mut cache = cache::CacheStore::new(tabby_common::path::cache_dir());
        index::index_repository(&mut cache, repository);
    }

    pub fn garbage_collection(&mut self, repositories: &[RepositoryConfig]) {
        self.is_dirty = false;
        let mut cache = cache::CacheStore::new(tabby_common::path::cache_dir());
        cache.garbage_collection_for_source_files();
        index::garbage_collection(&mut cache);
        repository::garbage_collection(repositories);
    }
}