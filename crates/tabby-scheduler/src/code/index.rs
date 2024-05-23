use ignore::Walk;
use kv::Batch;
use tabby_common::config::RepositoryConfig;
use tracing::warn;

use super::{cache::CacheStore, create_code_index, intelligence::SourceCode};
use crate::Indexer;

// Magic numbers
static MAX_LINE_LENGTH_THRESHOLD: usize = 300;
static AVG_LINE_LENGTH_THRESHOLD: f32 = 150f32;

pub async fn index_repository(cache: &mut CacheStore, repository: &RepositoryConfig) {
    let index = create_code_index();
    add_changed_documents(cache, repository, &index).await;
    index.commit();
}

pub fn garbage_collection(cache: &mut CacheStore) {
    let index = create_code_index();
    remove_staled_documents(cache, &index);
    index.commit();
}

async fn add_changed_documents(
    cache: &mut CacheStore,
    repository: &RepositoryConfig,
    index: &Indexer<SourceCode>,
) {
    let mut indexed_files_batch = Batch::new();
    for file in Walk::new(repository.dir()) {
        let file = match file {
            Ok(file) => file,
            Err(e) => {
                warn!("Failed to walk file tree for indexing: {e}");
                continue;
            }
        };
        let Some(source_file) = cache.get_source_file(repository, file.path()) else {
            continue;
        };
        if !is_valid_file(&source_file) {
            continue;
        }

        let (file_id, indexed) = cache.check_indexed(file.path());
        if indexed {
            continue;
        }
        index.add(source_file).await;
        indexed_files_batch
            .set(&file_id, &String::new())
            .expect("Failed to mark file as indexed");
    }

    // Mark all indexed documents as indexed
    cache.apply_indexed(indexed_files_batch);
}

fn remove_staled_documents(cache: &mut CacheStore, index: &Indexer<SourceCode>) {
    // Create a new writer to commit deletion of removed indexed files
    let gc_commit = cache.prepare_garbage_collection_for_indexed_files(|key| {
        index.delete(key);
    });

    gc_commit();
}

fn is_valid_file(file: &SourceCode) -> bool {
    file.max_line_length <= MAX_LINE_LENGTH_THRESHOLD
        && file.avg_line_length <= AVG_LINE_LENGTH_THRESHOLD
}
