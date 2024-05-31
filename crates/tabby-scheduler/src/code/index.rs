use std::sync::Arc;

use ignore::Walk;
use tabby_common::config::RepositoryConfig;
use tabby_inference::Embedding;
use tracing::warn;

use super::{
    cache::{CacheStore, IndexBatch},
    create_code_index,
    intelligence::SourceCode,
    KeyedSourceCode,
};
use crate::Indexer;

// Magic numbers
static MAX_LINE_LENGTH_THRESHOLD: usize = 300;
static AVG_LINE_LENGTH_THRESHOLD: f32 = 150f32;

pub async fn index_repository(
    cache: &mut CacheStore,
    embedding: Arc<dyn Embedding>,
    repository: &RepositoryConfig,
) {
    let index = create_code_index(Some(embedding));
    if index.recreated {
        cache.clear_indexed()
    }
    add_changed_documents(cache, repository, &index).await;
    index.commit();
}

pub fn garbage_collection(cache: &mut CacheStore) {
    let index = create_code_index(None);
    remove_staled_documents(cache, &index);
    index.commit();
}

async fn add_changed_documents(
    cache: &mut CacheStore,
    repository: &RepositoryConfig,
    index: &Indexer<KeyedSourceCode>,
) {
    let mut indexed_files_batch = IndexBatch::default();
    for file in Walk::new(repository.dir()) {
        let file = match file {
            Ok(file) => file,
            Err(e) => {
                warn!("Failed to walk file tree for indexing: {e}");
                continue;
            }
        };
        let Some(code) = cache.get_source_file(repository, file.path()) else {
            continue;
        };
        if !is_valid_file(&code) {
            continue;
        }

        let (key, indexed) = cache.check_indexed(file.path());
        if indexed {
            continue;
        }
        index
            .add(KeyedSourceCode {
                key: key.clone(),
                code,
            })
            .await;
        indexed_files_batch.set_indexed(key);
    }

    // Mark all indexed documents as indexed
    cache.apply_indexed(indexed_files_batch);
}

fn remove_staled_documents(cache: &mut CacheStore, index: &Indexer<KeyedSourceCode>) {
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

#[cfg(test)]
mod tests {
    use crate::code::intelligence::CodeIntelligence;
    use insta::assert_snapshot;

    #[test]
    fn test_code_splitter() {
        let intelligence = CodeIntelligence::default();
        // First file, chat/openai_chat.rs
        let file_contents = include_str!("../../../http-api-bindings/src/chat/openai_chat.rs");

        let rust_chunks = intelligence
            .chunks(file_contents, "rust")
            .map(|(_, chunk)| chunk)
            .collect::<Vec<_>>();

        assert_snapshot!(format!("{:#?}", rust_chunks));

        let text_chunks = intelligence
            .chunks(file_contents, "unknown")
            .map(|(_, chunk)| chunk)
            .collect::<Vec<_>>();

        assert_snapshot!(format!("{:#?}", text_chunks));

        // Second file, tabby-db/src/cache.rs
        let file_contents2 = include_str!("../../../../ee/tabby-db/src/cache.rs");

        let rust_chunks2 = intelligence
            .chunks(file_contents2, "rust")
            .map(|(_, chunk)| chunk)
            .collect::<Vec<_>>();

        assert_snapshot!(format!("{:#?}", rust_chunks2));

        let text_chunks2 = intelligence
            .chunks(file_contents2, "unknown")
            .map(|(_, chunk)| chunk)
            .collect::<Vec<_>>();

        assert_snapshot!(format!("{:#?}", text_chunks2));
    }
}
