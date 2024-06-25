use std::{pin::pin, sync::Arc};

use async_stream::stream;
use futures::StreamExt;
use ignore::{DirEntry, Walk};
use tabby_common::config::RepositoryConfig;
use tabby_inference::Embedding;
use tracing::warn;

use super::{
    create_code_index,
    intelligence::{CodeIntelligence, SourceCode},
};

// Magic numbers
static MAX_LINE_LENGTH_THRESHOLD: usize = 300;
static AVG_LINE_LENGTH_THRESHOLD: f32 = 150f32;

pub async fn index_repository(embedding: Arc<dyn Embedding>, repository: &RepositoryConfig) {
    let total_files = Walk::new(repository.dir()).count();
    let file_stream = stream! {
        for file in Walk::new(repository.dir()) {
            let file = match file {
                Ok(file) => file,
                Err(e) => {
                    warn!("Failed to walk file tree for indexing: {e}");
                    continue;
                }
            };

            yield file;
        }
    }
    // Commit every 200 files
    .chunks(200);

    let mut file_stream = pin!(file_stream);

    let mut count_files = 0;
    while let Some(files) = file_stream.next().await {
        count_files += files.len();
        add_changed_documents(repository, embedding.clone(), files).await;
        logkit::info!("{}/{} files has been processed...", count_files, total_files);
    }
}

pub async fn garbage_collection() {
    let index = create_code_index(None);
    stream! {
        let mut num_to_keep = 0;
        let mut num_to_delete = 0;

        for await id in index.iter_ids() {
            let item_key = id;
            if CodeIntelligence::check_source_file_id_matched(&item_key) {
                num_to_keep += 1;
            } else {
                num_to_delete += 1;
                index.delete(&item_key);
            }
        }

        logkit::info!("Finished garbage collection for code index: {num_to_keep} items kept, {num_to_delete} items removed");
        index.commit();
    }.collect::<()>().await;
}

async fn add_changed_documents(
    repository: &RepositoryConfig,
    embedding: Arc<dyn Embedding>,
    files: Vec<DirEntry>,
) {
    let index = Arc::new(create_code_index(Some(embedding)));
    let intelligence = CodeIntelligence::default();
    let cloned_index = index.clone();
    stream! {
        for file in files {
            let Some(key) = CodeIntelligence::compute_source_file_id(file.path()) else {
                continue;
            };

            if cloned_index.is_indexed(&key) {
                // Skip if already indexed
                continue;
            }

            let Some(code) = intelligence.compute_source_file(repository, file.path()) else {
                continue;
            };

            if !is_valid_file(&code) {
                continue;
            }

            let index = cloned_index.clone();

            // yield is lazy, that means only when the stream is polled, the task will be spawned.
            yield tokio::spawn(async move {
                index.add(code).await;
            });
        }
    }
    .buffer_unordered(std::cmp::max(
        std::thread::available_parallelism().unwrap().get() * 2,
        32,
    ))
    .map(|_| ())
    .collect::<()>()
    .await;

    match Arc::try_unwrap(index) {
        Ok(index) => index.commit(),
        Err(_) => {
            panic!("Failed to commit code index");
        }
    };
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
