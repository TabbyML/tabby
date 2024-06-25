//! Responsible for scheduling all of the background jobs for tabby.
//! Includes syncing respositories and updating indices.

mod code;
mod crawl;
mod indexer;

pub use code::CodeIndexer;
use crawl::crawl_pipeline;
use doc::create_web_index;
pub use doc::{DocIndexer, WebDocument};
use futures::{StreamExt};
use indexer::{IndexAttributeBuilder, Indexer};
use tabby_inference::Embedding;

mod doc;
use std::sync::Arc;

use tracing::{debug, info};

use crate::doc::SourceDocument;

pub async fn crawl_index_docs(
    urls: &[String],
    embedding: Arc<dyn Embedding>,
    on_process_url: impl Fn(String),
) -> anyhow::Result<()> {
    for url in urls {
        debug!("Starting doc index pipeline for {url}");
        let embedding = embedding.clone();
        let mut num_docs = 0;
        let doc_index = create_web_index(embedding.clone());

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
            doc_index.add(source_doc).await;
        }
        info!("Crawled {} documents from '{}'", num_docs, url);
        doc_index.commit();
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
