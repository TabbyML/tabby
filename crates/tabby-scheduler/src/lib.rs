//! Responsible for scheduling all of the background jobs for tabby.
//! Includes syncing respositories and updating indices.

mod code;
mod crawl;
mod indexer;

pub use code::CodeIndexer;
use crawl::crawl_pipeline;
use doc::create_web_builder;
pub use doc::{DocIndexer, WebDocument};
use futures::StreamExt;
use indexer::{IndexAttributeBuilder, Indexer};
use tabby_common::index::corpus;
use tabby_inference::Embedding;

mod doc;
use std::sync::Arc;

use crate::doc::SourceDocument;

pub async fn crawl_index_docs(
    source_id: &str,
    start_url: &str,
    embedding: Arc<dyn Embedding>,
) -> anyhow::Result<()> {
    logkit::info!("Starting doc index pipeline for {}", start_url);
    let embedding = embedding.clone();
    let mut num_docs = 0;
    let builder = create_web_builder(embedding.clone());
    let indexer = Indexer::new(corpus::WEB);

    let mut pipeline = Box::pin(crawl_pipeline(start_url).await?);
    while let Some(doc) = pipeline.next().await {
        logkit::info!("Fetching {}", doc.url);
        let source_doc = SourceDocument {
            source_id: source_id.to_owned(),
            id: doc.url.clone(),
            title: doc.metadata.title.unwrap_or_default(),
            link: doc.url,
            body: doc.markdown,
        };

        num_docs += 1;

        let (id, s) = builder.build(source_doc).await;
        indexer.delete(&id);
        s.buffer_unordered(std::cmp::max(
            std::thread::available_parallelism().unwrap().get() * 2,
            32,
        ))
        .for_each(|doc| async {
            if let Ok(Some(doc)) = doc {
                indexer.add(doc).await;
            }
        })
        .await;
    }
    logkit::info!("Crawled {} documents from '{}'", num_docs, start_url);
    indexer.commit();
    Ok(())
}

pub fn run_index_garbage_collection(active_sources: Vec<(String, String)>) -> anyhow::Result<()> {
    let corpus_list = [corpus::WEB, corpus::CODE];
    for corpus in corpus_list.iter() {
        let active_sources: Vec<_> = active_sources
            .iter()
            .filter(|(c, _)| c == corpus)
            .map(|(_, source_id)| source_id.to_owned())
            .collect();
        let indexer = Indexer::new(corpus);
        indexer.garbage_collect(&active_sources)?;
        indexer.commit();
    }

    Ok(())
}

mod tantivy_utils {
    use std::{fs, path::Path};

    use tantivy::{directory::MmapDirectory, schema::Schema, Index};
    use tracing::{debug, warn};

    pub fn open_or_create_index(code: &Schema, path: &Path) -> (bool, Index) {
        let (recreated, index) = match open_or_create_index_impl(code, path) {
            Ok(index) => (false, index),
            Err(err) => {
                warn!(
                    "Failed to open index repositories: {}, removing index directory '{}'...",
                    err,
                    path.display()
                );
                fs::remove_dir_all(path).expect("Failed to remove index directory");

                debug!("Reopening index repositories...");
                (
                    true,
                    open_or_create_index_impl(code, path).expect("Failed to open index"),
                )
            }
        };
        (recreated, index)
    }

    fn open_or_create_index_impl(code: &Schema, path: &Path) -> tantivy::Result<Index> {
        fs::create_dir_all(path).expect("Failed to create index directory");
        let directory = MmapDirectory::open(path).expect("Failed to open index directory");
        Index::open_or_create(directory, code.clone())
    }
}
