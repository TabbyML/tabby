use std::{env, pin::pin, sync::Arc};

use async_stream::stream;
use futures::StreamExt;
use ignore::Walk;
use tabby_common::config::RepositoryConfig;
use tabby_inference::Embedding;
use tracing::{debug, info, warn};

use super::{
    cache::{source_file_key_from_path, CacheStore, IndexBatch},
    create_code_index,
    intelligence::{CodeIntelligence, SourceCode},
    KeyedSourceCode,
};
use crate::{code::cache::is_item_key_matched, Indexer};

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
    let index_batch = add_changed_documents(repository, index).await;
    cache.apply_indexed(index_batch);
}

pub async fn garbage_collection(cache: &mut CacheStore) {
    let index = create_code_index(None);
    remove_staled_documents(&index).await;
    index.commit();
}

async fn add_changed_documents(
    repository: &RepositoryConfig,
    index: Indexer<KeyedSourceCode>,
) -> IndexBatch {
    let index = Arc::new(index);
    let cloned_index = index.clone();
    let s = stream! {
        let mut intelligence = CodeIntelligence::default();
        for file in Walk::new(repository.dir()) {
            let file = match file {
                Ok(file) => file,
                Err(e) => {
                    warn!("Failed to walk file tree for indexing: {e}");
                    continue;
                }
            };

            let Some(key) = source_file_key_from_path(file.path()) else {
                continue;
            };

            if cloned_index.is_id_indexed(&key) {
                continue;
            }

            let Some(code) = intelligence.create_source_file(repository, file.path()) else {
                continue;
            };

            if !is_valid_file(&code) {
                continue;
            }

            let index = cloned_index.clone();
            yield tokio::spawn(async move {
                index
                    .add(KeyedSourceCode {
                        key: key.clone(),
                        code,
                    })
                    .await;
                key
            });
        }

    };

    let parallelism = env::var("TABBY_CODE_INDEXER_PARALLELISM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().unwrap().get() * 2);
    let mut s = pin!(s.buffer_unordered(parallelism));

    let mut indexed_files_batch = IndexBatch::default();
    while let Some(key) = s.next().await {
        if let Ok(key) = key {
            indexed_files_batch.set_indexed(key);
        } else {
            warn!("Failed to index file");
        }
    }

    match Arc::try_unwrap(index) {
        Ok(index) => index.commit(),
        Err(_) => panic!("Failed to unwrap index"),
    }

    indexed_files_batch
}

async fn remove_staled_documents(index: &Indexer<KeyedSourceCode>) {
    stream! {
        let mut num_to_keep = 0;
        let mut num_to_delete = 0;

        for await id in index.iter_ids() {
            let item_key = id;
            if is_item_key_matched(&item_key) {
                num_to_keep += 1;
            } else {
                num_to_delete += 1;
                index.delete(&item_key);
            }
        }

        info!("Finished garbage collection for code index: {num_to_keep} items kept, {num_to_delete} items removed");
    }.collect::<()>().await;
}

fn is_valid_file(file: &SourceCode) -> bool {
    file.max_line_length <= MAX_LINE_LENGTH_THRESHOLD
        && file.avg_line_length <= AVG_LINE_LENGTH_THRESHOLD
}

#[cfg(test)]
mod tests {
    use insta::assert_snapshot;

    use crate::code::intelligence::CodeIntelligence;

    #[test]
    fn test_code_splitter() {
        let intelligence = CodeIntelligence::default();
        // First file, chat/openai_chat.rs
        let file_contents = include_str!("../../../http-api-bindings/src/chat/openai_chat.rs");

        let rust_chunks = intelligence
            .chunks(file_contents, "rust")
            .into_iter()
            .map(|(_, chunk)| chunk)
            .collect::<Vec<_>>();

        assert_snapshot!(format!("{:#?}", rust_chunks));

        let text_chunks = intelligence
            .chunks(file_contents, "unknown")
            .into_iter()
            .map(|(_, chunk)| chunk)
            .collect::<Vec<_>>();

        assert_snapshot!(format!("{:#?}", text_chunks));

        // Second file, tabby-db/src/cache.rs
        let file_contents2 = include_str!("../../../../ee/tabby-db/src/cache.rs");

        let rust_chunks2 = intelligence
            .chunks(file_contents2, "rust")
            .into_iter()
            .map(|(_, chunk)| chunk)
            .collect::<Vec<_>>();

        assert_snapshot!(format!("{:#?}", rust_chunks2));

        let text_chunks2 = intelligence
            .chunks(file_contents2, "unknown")
            .into_iter()
            .map(|(_, chunk)| chunk)
            .collect::<Vec<_>>();

        assert_snapshot!(format!("{:#?}", text_chunks2));
    }
}
