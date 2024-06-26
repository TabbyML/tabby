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
    urls: &[String],
    embedding: Arc<dyn Embedding>,
    on_process_url: impl Fn(String),
) -> anyhow::Result<()> {
    for url in urls {
        logkit::info!("Starting doc index pipeline for {url}");
        let embedding = embedding.clone();
        let mut num_docs = 0;
        let builder = create_web_builder(embedding.clone());
        let indexer = Indexer::new(corpus::WEB);

        let mut pipeline = Box::pin(crawl_pipeline(url).await?);
        while let Some(doc) = pipeline.next().await {
            on_process_url(doc.url.clone());
            let source_doc = SourceDocument {
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
        logkit::info!("Crawled {} documents from '{}'", num_docs, url);
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
