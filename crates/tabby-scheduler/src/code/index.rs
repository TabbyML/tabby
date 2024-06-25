use std::sync::Arc;

use async_stream::stream;
use futures::StreamExt;
use ignore::Walk;
use tabby_common::config::RepositoryConfig;
use tabby_inference::Embedding;
use tracing::warn;

use super::{
    create_code_index,
    intelligence::{CodeIntelligence, SourceCode},
};
use crate::Indexer;

// Magic numbers
static MAX_LINE_LENGTH_THRESHOLD: usize = 300;
static AVG_LINE_LENGTH_THRESHOLD: f32 = 150f32;

pub async fn index_repository(embedding: Arc<dyn Embedding>, repository: &RepositoryConfig) {
    let index = create_code_index(Some(embedding));
    add_changed_documents(repository, index).await;
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

async fn add_changed_documents(repository: &RepositoryConfig, index: Indexer<SourceCode>) {
    let index = Arc::new(index);
    let intelligence = CodeIntelligence::default();
    for file in Walk::new(repository.dir()) {
        let file = match file {
            Ok(file) => file,
            Err(e) => {
                warn!("Failed to walk file tree for indexing: {e}");
                continue;
            }
        };

        let Some(key) = CodeIntelligence::compute_source_file_id(file.path()) else {
            continue;
        };

        if index.is_indexed(&key) {
            // Skip if already indexed
            continue;
        }

        logkit::info!("Indexing file: {}", file.path().display());

        let Some(code) = intelligence.compute_source_file(repository, file.path()) else {
            continue;
        };

        if !is_valid_file(&code) {
            continue;
        }

        let index = index.clone();
        tokio::spawn(async move {
            index.add(code).await;
        });
    }

    wait_for_index(index).await;
}

async fn wait_for_index(index: Arc<Indexer<SourceCode>>) {
    let mut current_index = Box::new(index);
    loop {
        match Arc::try_unwrap(*current_index) {
            Ok(index) => {
                index.commit();
                break;
            }
            Err(index) => {
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                *current_index = index;
            }
        }
    }
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
